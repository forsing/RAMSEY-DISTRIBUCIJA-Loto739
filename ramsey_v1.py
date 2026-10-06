# Potreban je NumPy ≥ 2.0

import argparse
import csv
import itertools
import math
import os
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np


DEFAULT_CSV = "/data/loto7_4696_k79.csv"
# DEFAULT_CSV = "/data/loto7_4696_k79_loto_2970.csv"
# DEFAULT_CSV = "/data/loto7_4696_k79_loto_plus_1726.csv"

PAIRS = np.array(list(itertools.combinations(range(7), 2)))
TRIPLES = np.array(list(itertools.combinations(range(7), 3)))
TOTAL_COMBINATIONS = math.comb(39, 7)


def read_csv(path):
    """CSV bez zaglavlja: sedam brojeva po redu."""
    rows = []

    with open(path, encoding="utf-8-sig", newline="") as f:
        for line, row in enumerate(csv.reader(f), start=1):
            if not row or all(not value.strip() for value in row):
                continue

            try:
                numbers = sorted(int(value.strip()) for value in row)
            except ValueError as error:
                raise ValueError(
                    f"Red {line}: očekujem sedam celih brojeva."
                ) from error

            if (
                len(numbers) != 7
                or len(set(numbers)) != 7
                or numbers[0] < 1
                or numbers[-1] > 39
            ):
                raise ValueError(
                    f"Red {line}: neispravna kombinacija {numbers}."
                )

            rows.append(numbers)

    if len(rows) < 100:
        raise ValueError("Potrebno je najmanje 100 kola.")

    return np.asarray(rows, dtype=np.float64)


def distance_threshold(draws):
    """Medijana distribucije svih unutrašnjih rastojanja."""
    distances = (
        draws[:, PAIRS[:, 1]]
        - draws[:, PAIRS[:, 0]]
    )
    return float(np.median(distances))


def features(draws, threshold):
    """
    Jedanaest osobina kombinacije:
      - sedam sortiranih pozicija;
      - entropija osam razmaka, uključujući granice 0 i 40;
      - standardna devijacija razmaka;
      - udeo trouglova sa sve tri duge veze;
      - udeo trouglova sa sve tri kratke veze.

    Bojenje grafa zavisi od rastojanja, ne od učestalosti brojeva.
    """
    boundaries = np.column_stack(
        (
            np.zeros(len(draws)),
            draws,
            np.full(len(draws), 40.0),
        )
    )
    gaps = np.diff(boundaries, axis=1)

    proportions = gaps / 40.0
    entropy = -np.sum(
        proportions * np.log(proportions),
        axis=1,
    )
    spread = np.std(gaps, axis=1)

    outer_long = (
        draws[:, TRIPLES[:, 2]]
        - draws[:, TRIPLES[:, 0]]
    ) > threshold

    left_long = (
        draws[:, TRIPLES[:, 1]]
        - draws[:, TRIPLES[:, 0]]
    ) > threshold

    right_long = (
        draws[:, TRIPLES[:, 2]]
        - draws[:, TRIPLES[:, 1]]
    ) > threshold

    long_triangles = np.sum(
        outer_long & left_long & right_long,
        axis=1,
    ) / 35.0

    short_triangles = np.sum(
        ~outer_long & ~left_long & ~right_long,
        axis=1,
    ) / 35.0

    return np.column_stack(
        (
            draws,
            entropy,
            spread,
            long_triangles,
            short_triangles,
        )
    )


def distribution_scaler(values):
    """Robusna normalizacija pomoću medijane i kvartila."""
    center = np.median(values, axis=0)
    quartiles = np.percentile(values, [75, 25], axis=0)
    scale = (quartiles[0] - quartiles[1]) / 1.349

    scale = np.where(
        scale > 1e-9,
        scale,
        np.std(values, axis=0),
    )
    return center, np.maximum(scale, 1e-6)


def fit_distribution(values, alpha):
    """
    alpha=None:
        Zajednička stacionarna Gaussova aproksimacija distribucije.

    alpha>0:
        Uslovna distribucija sledećeg kola, sa ridge regresijom
        nad osobinama prethodnog kola.
    """
    dimension = values.shape[1]

    if alpha is None:
        mean = np.mean(values, axis=0)
        coefficients = None
        residuals = values - mean

    else:
        previous = values[:-1]
        following = values[1:]

        previous_mean = previous.mean(axis=0)
        following_mean = following.mean(axis=0)

        x = previous - previous_mean
        y = following - following_mean

        coefficients = np.linalg.solve(
            x.T @ x + alpha * np.eye(dimension),
            x.T @ y,
        )

        mean = (previous_mean, following_mean)

        fitted = following_mean + x @ coefficients
        residuals = following - fitted

    covariance = residuals.T @ residuals / len(residuals)

    # Stabilizacija kovarijanse zbog povezanih strukturnih osobina.
    covariance = (
        0.95 * covariance
        + 0.05 * np.diag(np.diag(covariance))
        + 1e-6 * np.eye(dimension)
    )

    precision = np.linalg.inv(covariance)
    sign, log_determinant = np.linalg.slogdet(covariance)

    if sign <= 0:
        raise ArithmeticError("Kovarijansa nije pozitivno definitna.")

    return mean, coefficients, precision, log_determinant


def predict_center(model, previous):
    mean, coefficients, _, _ = model

    if coefficients is None:
        return mean

    previous_mean, following_mean = mean

    return (
        following_mean
        + (previous - previous_mean) @ coefficients
    )


def select_model(draws):
    """
    Hronološka provera:
      prvih 80% kola za učenje;
      poslednjih 20% za ocenu.

    Prag, normalizacija i parametri tokom provere nastaju
    isključivo iz dela za učenje.
    """
    split = int(0.8 * len(draws))
    training_draws = draws[:split]

    threshold = distance_threshold(training_draws)
    raw = features(draws, threshold)

    center, scale = distribution_scaler(raw[:split])
    normalized = (raw - center) / scale

    results = []

    for alpha in (None, 1.0, 10.0, 100.0, 1000.0):
        model = fit_distribution(normalized[:split], alpha)

        predicted = predict_center(
            model,
            normalized[split - 1:-1],
        )
        differences = normalized[split:] - predicted

        quadratic = np.einsum(
            "bi,ij,bj->b",
            differences,
            model[2],
            differences,
        )

        losses = 0.5 * (
            quadratic
            + model[3]
            + normalized.shape[1] * math.log(2.0 * math.pi)
        )

        results.append((float(losses.mean()), alpha))

    # Pri jednakom skoru prednost ima stacionarni model,
    # zatim jača regularizacija.
    best = min(
        results,
        key=lambda item: (
            item[0],
            item[1] is not None,
            -(item[1] or 0.0),
        ),
    )

    return best[1], results, split


def exhaustive_prediction(draws, alpha, batch_size):
    """
    Konačno učenje koristi ceo CSV.
    Pretraga obuhvata svaku moguću kombinaciju tačno jednom.
    """
    threshold = distance_threshold(draws)
    raw = features(draws, threshold)

    center, scale = distribution_scaler(raw)
    normalized = (raw - center) / scale

    model = fit_distribution(normalized, alpha)
    target = predict_center(model, normalized[-1])

    candidates = itertools.combinations(range(1, 40), 7)

    best_score = math.inf
    best_combination = None
    checked = 0
    next_report = 1_000_000

    while True:
        block = list(itertools.islice(candidates, batch_size))

        if not block:
            break

        numbers = np.asarray(block, dtype=np.float64)
        candidate_features = features(numbers, threshold)

        differences = (
            (candidate_features - center) / scale
            - target
        )

        scores = np.einsum(
            "bi,ij,bj->b",
            differences,
            model[2],
            differences,
            optimize=True,
        )

        index = int(np.argmin(scores))
        score = float(scores[index])

        # Leksikografski redosled rešava eventualne jednake skorove.
        if score < best_score:
            best_score = score
            best_combination = block[index]

        checked += len(block)

        if checked >= next_report:
            print(
                f"Provereno {checked:,}/{TOTAL_COMBINATIONS:,}",
                file=sys.stderr,
                flush=True,
            )
            next_report += 1_000_000

    if checked != TOTAL_COMBINATIONS:
        raise ArithmeticError("Pretraga nije obuhvatila sve kombinacije.")

    return best_combination, best_score, threshold, checked


def model_name(alpha):
    if alpha is None:
        return "stacionarni"
    return f"uslovni, alpha={alpha:g}"


def main():
    parser = argparse.ArgumentParser(
        description="Ramsey-distribucija loto 7/39 — v1"
    )
    parser.add_argument(
        "csv",
        nargs="?",
        default=DEFAULT_CSV,
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=20_000,
        help="Broj kombinacija u memoriji po koraku.",
    )
    args = parser.parse_args()

    if args.batch < 1:
        parser.error("--batch mora biti pozitivan.")

    draws = read_csv(args.csv)
    alpha, results, split = select_model(draws)

    print("RAMSEY-DISTRIBUCIJA V1")
    print(f"CSV: {args.csv}")
    print(f"Broj kola: {len(draws)}")
    print(
        f"Hronološka provera: {split} kola za učenje, "
        f"{len(draws) - split} za ocenu."
    )

    for loss, candidate in results:
        print(
            f"  {model_name(candidate)}: "
            f"negativni log-skor = {loss:.6f}"
        )

    print(f"Izabrani model: {model_name(alpha)}")
    print("Konačno učenje na celom CSV-u.")
    print(
        f"Pretraga svih {TOTAL_COMBINATIONS:,} kombinacija.",
        flush=True,
    )

    combination, score, threshold, checked = exhaustive_prediction(
        draws,
        alpha,
        args.batch,
    )

    print(f"\nPrag bojenja rastojanja: {threshold:g}")
    print(f"Provereno kombinacija: {checked}")
    print("NEXT:", " ".join(f"{number:02d}" for number in combination))
    print(f"Distribucioni skor: {score:.9f}")
    print("Skor je kriterijum modela, nije verovatnoća izvlačenja.")


if __name__ == "__main__":
    main()



"""
RAMSEY-DISTRIBUCIJA V1
CSV: /data/loto7_4696_k79.csv
Broj kola: 4696
Hronološka provera: 3756 kola za učenje, 940 za ocenu.
  stacionarni: negativni log-skor = 11.232880
  uslovni, alpha=1: negativni log-skor = 11.249206
  uslovni, alpha=10: negativni log-skor = 11.248820
  uslovni, alpha=100: negativni log-skor = 11.246497
  uslovni, alpha=1000: negativni log-skor = 11.240340
Izabrani model: stacionarni
Konačno učenje na celom CSV-u.
Pretraga svih 15,380,937 kombinacija.
Provereno 1,000,000/15,380,937
Provereno 2,000,000/15,380,937
Provereno 3,000,000/15,380,937
Provereno 4,000,000/15,380,937
Provereno 5,000,000/15,380,937
Provereno 6,000,000/15,380,937
Provereno 7,000,000/15,380,937
Provereno 8,000,000/15,380,937
Provereno 9,000,000/15,380,937
Provereno 10,000,000/15,380,937
Provereno 11,000,000/15,380,937
Provereno 12,000,000/15,380,937
Provereno 13,000,000/15,380,937
Provereno 14,000,000/15,380,937
Provereno 15,000,000/15,380,937

Prag bojenja rastojanja: 12
Provereno kombinacija: 15380937
NEXT: 05 x 13 y 23 z 36
Distribucioni skor: 3.559873906
Skor je kriterijum modela, nije verovatnoća izvlačenja.





RAMSEY-DISTRIBUCIJA V1
CSV: /data/loto7_4696_k79_loto_2970.csv
Broj kola: 2970
Hronološka provera: 2376 kola za učenje, 594 za ocenu.
  stacionarni: negativni log-skor = 11.144005
  uslovni, alpha=1: negativni log-skor = 11.168918
  uslovni, alpha=10: negativni log-skor = 11.167900
  uslovni, alpha=100: negativni log-skor = 11.163692
  uslovni, alpha=1000: negativni log-skor = 11.154969
Izabrani model: stacionarni
Konačno učenje na celom CSV-u.
Pretraga svih 15,380,937 kombinacija.
Provereno 1,000,000/15,380,937
Provereno 2,000,000/15,380,937
Provereno 3,000,000/15,380,937
Provereno 4,000,000/15,380,937
Provereno 5,000,000/15,380,937
Provereno 6,000,000/15,380,937
Provereno 7,000,000/15,380,937
Provereno 8,000,000/15,380,937
Provereno 9,000,000/15,380,937
Provereno 10,000,000/15,380,937
Provereno 11,000,000/15,380,937
Provereno 12,000,000/15,380,937
Provereno 13,000,000/15,380,937
Provereno 14,000,000/15,380,937
Provereno 15,000,000/15,380,937

Prag bojenja rastojanja: 12
Provereno kombinacija: 15380937
NEXT: 07 x 13 y 24 z 35
Distribucioni skor: 3.503504985
Skor je kriterijum modela, nije verovatnoća izvlačenja.





RAMSEY-DISTRIBUCIJA V1
CSV: /data/loto7_4696_k79_loto_plus_1726.csv
Broj kola: 1726
Hronološka provera: 1380 kola za učenje, 346 za ocenu.
  stacionarni: negativni log-skor = 10.906058
  uslovni, alpha=1: negativni log-skor = 10.937314
  uslovni, alpha=10: negativni log-skor = 10.935001
  uslovni, alpha=100: negativni log-skor = 10.925342
  uslovni, alpha=1000: negativni log-skor = 10.910409
Izabrani model: stacionarni
Konačno učenje na celom CSV-u.
Pretraga svih 15,380,937 kombinacija.
Provereno 1,000,000/15,380,937
Provereno 2,000,000/15,380,937
Provereno 3,000,000/15,380,937
Provereno 4,000,000/15,380,937
Provereno 5,000,000/15,380,937
Provereno 6,000,000/15,380,937
Provereno 7,000,000/15,380,937
Provereno 8,000,000/15,380,937
Provereno 9,000,000/15,380,937
Provereno 10,000,000/15,380,937
Provereno 11,000,000/15,380,937
Provereno 12,000,000/15,380,937
Provereno 13,000,000/15,380,937
Provereno 14,000,000/15,380,937
Provereno 15,000,000/15,380,937

Prag bojenja rastojanja: 12
Provereno kombinacija: 15380937
NEXT: 05 x 18 y 28 z 36
Distribucioni skor: 3.568669489
Skor je kriterijum modela, nije verovatnoća izvlačenja.
"""



"""
hipoteza inspirisana Ramzijevom teorijom

Ramzijeva teorija kaže da se u dovoljno velikim strukturama, 
pod određenim uslovima, nužno pojavljuju pravilne podstrukture 
— čak i kada je raspored nasumičan. 
Ona garantuje postojanje određenih obrazaca, 
ali ne govori kada će se oni pojaviti u budućim izvlačenjima. 


Koristan pristup je da brojeve 1–39 predstavim kao čvorove grafa, 
a njihove zajedničke pojave kao veze. 
Veze označim prema tome koliko su učestalije ili redje nego što očekujem. 
Onda tražim grupe brojeva sa medjusobno sličnim vezama 
— strukture inspirisane Ramzijevom teorijom 
— i koristim ih za rangiranje kandidata za naredno kolo.


Koristi zajedničku distribuciju pozicija, razmaka i jednobojnih trouglova u grafu rastojanja. 
Proverava svih 15.380.937 kombinacija i ispisuje jednu NEXT kombinaciju.
Redosled CSV-a tretira kao od najstarijeg ka najnovijem kolu. 


Na mom CSV-u provera je izabrala stacionarnu distribuciju: 
uslovljavanje prethodnim kolom imalo je slabiji rezultat na poslednjih 940 kola. 
Konačan izbor koristi svih 4.696 redova.
"""
