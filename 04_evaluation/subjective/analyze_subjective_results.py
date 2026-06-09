#!/usr/bin/env python3
"""Analiza respuestas MOS o CMOS exportadas a CSV.

MOS resume columnas numéricas como naturalidad, inteligibilidad y calidad.
CMOS usa una columna normalizada para que valores positivos favorezcan
DDPM+BETO. Solo requiere la biblioteca estándar de Python.
"""

import argparse
import csv
import math
import statistics
from pathlib import Path


MOS_COLUMNS = ("naturalidad", "inteligibilidad", "calidad")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_file", help="CSV de respuestas.")
    parser.add_argument("--mode", choices=("mos", "cmos"), required=True)
    parser.add_argument(
        "--score-column",
        help="Columna CMOS. Si se omite, se usa la primera columna numérica.",
    )
    return parser.parse_args()


def read_csv(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        if not reader.fieldnames:
            raise ValueError("El CSV no contiene encabezados")
    if not rows:
        raise ValueError("El CSV no contiene respuestas")
    return reader.fieldnames, rows


def numeric_values(rows, column):
    values = []
    for row in rows:
        raw = (row.get(column) or "").strip().replace(",", ".")
        if not raw:
            continue
        try:
            values.append(float(raw))
        except ValueError:
            continue
    return values


def summarize(values):
    if not values:
        raise ValueError("No hay valores numéricos para resumir")
    deviation = statistics.stdev(values) if len(values) > 1 else float("nan")
    ic95 = (
        1.96 * deviation / math.sqrt(len(values))
        if len(values) > 1 else float("nan")
    )
    return {
        "N": len(values),
        "media": statistics.fmean(values),
        "desviacion": deviation,
        "mediana": statistics.median(values),
        "IC95": ic95,
    }


def numeric_columns(fieldnames, rows):
    return [
        column for column in fieldnames
        if numeric_values(rows, column)
    ]


def print_summary(label, stats):
    print(f"\n{label}")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")


def analyze_mos(fieldnames, rows):
    available = numeric_columns(fieldnames, rows)
    preferred = [
        column for expected in MOS_COLUMNS
        for column in available if column.lower() == expected
    ]
    columns = preferred or available
    if not columns:
        raise ValueError("El CSV MOS no contiene columnas numéricas")
    for column in columns:
        values = numeric_values(rows, column)
        if not all(1 <= value <= 5 for value in values):
            raise ValueError(f"Los puntajes MOS de {column} deben estar entre 1 y 5")
        print_summary(column, summarize(values))


def analyze_cmos(fieldnames, rows, score_column):
    available = numeric_columns(fieldnames, rows)
    if score_column and score_column not in fieldnames:
        raise ValueError(f"No existe la columna: {score_column}")
    column = score_column or (available[0] if available else None)
    if column is None:
        raise ValueError("El CSV CMOS no contiene una columna numérica")

    scores = numeric_values(rows, column)
    if not all(-3 <= score <= 3 for score in scores):
        raise ValueError("Los puntajes CMOS deben estar entre -3 y +3")

    print_summary(f"CMOS ({column})", summarize(scores))
    total = len(scores)
    print("\nDistribución de preferencias (%)")
    print(f"  DDPM+BETO: {100 * sum(score > 0 for score in scores) / total:.2f}")
    print(f"  Sin preferencia: {100 * sum(score == 0 for score in scores) / total:.2f}")
    print(f"  CoquiTTS: {100 * sum(score < 0 for score in scores) / total:.2f}")


def main() -> None:
    args = parse_args()
    fieldnames, rows = read_csv(Path(args.csv_file))
    if args.mode == "mos":
        analyze_mos(fieldnames, rows)
    else:
        analyze_cmos(fieldnames, rows, args.score_column)


if __name__ == "__main__":
    main()
