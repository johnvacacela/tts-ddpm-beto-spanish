#!/usr/bin/env python3
"""Genera filelists reproducibles para Baseline FastPitch y DDPM+BETO.

Se construye un unico split estratificado por dialecto. Ambos modelos reciben
exactamente las mismas muestras:

* Baseline: ``wav_path|text_orig|ipa_text``
* DDPM+BETO: ``wav_path|text_orig|ipa_text|text_orig``

La cuarta columna se conserva por compatibilidad con el cargador BETO.
"""

import argparse
import random
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", default="data/metadata.txt")
    parser.add_argument("--out_dir", default="filelists")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--val_ratio", type=float, default=0.10)
    return parser.parse_args()


def parse_metadata(path: Path):
    records = []
    with path.open(encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) != 3 or not all(parts):
                raise ValueError(
                    f"Linea {line_number} invalida en {path}; "
                    "se esperaban 3 columnas no vacias"
                )
            records.append(tuple(parts))
    return records


def dialect_code(wav_path: str) -> str:
    prefix = Path(wav_path).stem.split("_", maxsplit=1)[0].lower()
    return prefix if prefix in {"ar", "co", "pe", "ve", "cl"} else "unknown"


def stratified_split(records, val_ratio: float, seed: int):
    groups = defaultdict(list)
    for record in records:
        groups[dialect_code(record[0])].append(record)

    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)

    target_val = round(len(records) * val_ratio)
    quotas = {
        code: len(group) * val_ratio for code, group in groups.items()
    }
    allocations = {code: int(quota) for code, quota in quotas.items()}
    remaining = target_val - sum(allocations.values())
    remainders = sorted(
        groups,
        key=lambda code: (quotas[code] - allocations[code], code),
        reverse=True,
    )
    for code in remainders[:remaining]:
        allocations[code] += 1

    train, val = [], []
    for code in sorted(groups):
        n_val = allocations[code]
        val.extend(groups[code][:n_val])
        train.extend(groups[code][n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def write_filelist(path: Path, records, include_beto_text: bool) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for wav_path, text, ipa in records:
            columns = [wav_path, text, ipa]
            if include_beto_text:
                columns.append(text)
            file.write("|".join(columns) + "\n")


def main() -> None:
    args = parse_args()
    metadata = Path(args.metadata).expanduser()
    out_dir = Path(args.out_dir).expanduser()

    if not 0.0 < args.val_ratio < 1.0:
        raise ValueError("--val_ratio debe estar entre 0 y 1")
    if not metadata.is_file():
        raise FileNotFoundError(f"No existe metadata: {metadata}")

    records = parse_metadata(metadata)
    if not records:
        raise RuntimeError("Metadata no contiene muestras validas")

    train, val = stratified_split(records, args.val_ratio, args.seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs = (
        ("slr_all_train.txt", train, False),
        ("slr_all_val.txt", val, False),
        ("slr_bert_train.txt", train, True),
        ("slr_bert_val.txt", val, True),
    )
    for filename, split, include_beto_text in outputs:
        path = out_dir / filename
        write_filelist(path, split, include_beto_text)
        print(f"Generado: {path}")

    total = len(records)
    print(f"Total: {total}")
    print(f"Train: {len(train)} ({100 * len(train) / total:.2f}%)")
    print(f"Val: {len(val)} ({100 * len(val) / total:.2f}%)")
    print(f"Seed: {args.seed}")


if __name__ == "__main__":
    main()
