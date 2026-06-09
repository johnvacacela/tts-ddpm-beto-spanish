#!/usr/bin/env python3
"""Valida tamano, balance, longitudes y cobertura fonetica del split.

La prueba Kolmogorov-Smirnov (KS) comprueba si train y validacion presentan
distribuciones de longitud de frase similares; un valor p > 0.05 indica que
no se detecta una diferencia estadisticamente significativa.
"""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp


DIALECT_NAMES = {
    "ar": "Argentina",
    "co": "Colombia",
    "pe": "Peru",
    "ve": "Venezuela",
    "cl": "Chile",
}
IPA_SYMBOLS = {
    "p", "b", "t", "d", "k", "ɡ", "tʃ", "dʒ", "f", "β", "s", "z",
    "ʃ", "x", "θ", "ð", "ɾ", "r", "m", "n", "ɲ", "ŋ", "l", "ʎ",
    "j", "w", "ɣ", "i", "e", "a", "o", "u", "ɛ", "ɪ", "ʊ", "ɔ",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_file", default="filelists/slr_all_train.txt")
    parser.add_argument("--val_file", default="filelists/slr_all_val.txt")
    return parser.parse_args()


def read_filelist(path: Path):
    records = []
    with path.open(encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) < 3:
                raise ValueError(f"Formato invalido en {path}:{line_number}")
            records.append(parts)
    return records


def dialect_of(record) -> str:
    code = Path(record[0]).stem.split("_", maxsplit=1)[0].lower()
    return code if code in DIALECT_NAMES else "unknown"


def phoneme_inventory(records):
    inventory = set()
    for record in records:
        ipa = record[2].replace("ˈ", "").replace("ˌ", "")
        inventory.update(symbol for symbol in IPA_SYMBOLS if symbol in ipa)
    return inventory


def length_summary(lengths) -> str:
    return (
        f"media={np.mean(lengths):.2f}, std={np.std(lengths):.2f}, "
        f"min={min(lengths)}, max={max(lengths)}"
    )


def main() -> None:
    args = parse_args()
    train_file = Path(args.train_file).expanduser()
    val_file = Path(args.val_file).expanduser()
    for path in (train_file, val_file):
        if not path.is_file():
            raise FileNotFoundError(f"No existe el filelist: {path}")

    train = read_filelist(train_file)
    val = read_filelist(val_file)
    if not train or not val:
        raise RuntimeError("Train y val deben contener al menos una muestra")

    total = len(train) + len(val)
    train_pct = 100 * len(train) / total
    val_pct = 100 * len(val) / total

    print("=" * 64)
    print("VALIDACION DEL SPLIT TRAIN/VAL")
    print("=" * 64)
    print(f"Total: {total:,}")
    print(f"Train: {len(train):,} ({train_pct:.2f}%)")
    print(f"Val:   {len(val):,} ({val_pct:.2f}%)")

    train_dialects = Counter(map(dialect_of, train))
    val_dialects = Counter(map(dialect_of, val))
    dialect_ok = True
    print("\n1. Balance aproximado por dialecto")
    for code in sorted(set(train_dialects) | set(val_dialects)):
        train_count = train_dialects[code]
        val_count = val_dialects[code]
        dialect_total = train_count + val_count
        dialect_val_pct = 100 * val_count / dialect_total
        ok = abs(dialect_val_pct - val_pct) <= 2.0
        dialect_ok &= ok
        name = DIALECT_NAMES.get(code, f"Desconocido ({code})")
        print(
            f"  [{('OK' if ok else 'REVISAR'):7}] {name:12} "
            f"train={train_count:5} val={val_count:4} "
            f"val={dialect_val_pct:5.2f}%"
        )

    train_lengths = [len(record[2].split()) for record in train]
    val_lengths = [len(record[2].split()) for record in val]
    ks_stat, ks_pvalue = ks_2samp(train_lengths, val_lengths)
    ks_ok = ks_pvalue > 0.05
    print("\n2. Distribucion de longitud (tokens IPA)")
    print(f"  Train: {length_summary(train_lengths)}")
    print(f"  Val:   {length_summary(val_lengths)}")
    print(
        f"  KS: estadistico={ks_stat:.4f}, p={ks_pvalue:.4f} "
        f"[{('SIMILARES' if ks_ok else 'REVISAR')}]"
    )

    train_phonemes = phoneme_inventory(train)
    val_phonemes = phoneme_inventory(val)
    missing_in_val = train_phonemes - val_phonemes
    coverage = (
        100 * len(train_phonemes & val_phonemes) / len(train_phonemes)
        if train_phonemes else 100.0
    )
    phonetic_ok = not missing_in_val
    print("\n3. Cobertura fonetica de val respecto a train")
    print(f"  Inventario train: {len(train_phonemes)} simbolos")
    print(f"  Inventario val:   {len(val_phonemes)} simbolos")
    print(f"  Cobertura:        {coverage:.2f}%")
    if missing_in_val:
        print(f"  Ausentes en val:  {sorted(missing_in_val)}")

    ratio_ok = abs(val_pct - 10.0) <= 0.5
    checks = {
        "Split cercano a 90/10": ratio_ok,
        "Balance por dialecto": dialect_ok,
        "Longitudes similares (KS)": ks_ok,
        "Cobertura fonetica completa": phonetic_ok,
    }
    print("\n" + "=" * 64)
    print("RESUMEN")
    print("=" * 64)
    for label, ok in checks.items():
        print(f"[{('OK' if ok else 'REVISAR'):7}] {label}")
    print(
        "\nResultado: "
        + ("split validado" if all(checks.values()) else "requiere revision")
    )


if __name__ == "__main__":
    main()
