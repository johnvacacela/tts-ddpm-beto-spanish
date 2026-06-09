#!/usr/bin/env python3
"""Organiza OpenSLR 61, remuestrea audio y genera transcripciones IPA.

La salida ``metadata.txt`` usa el formato:
``wav_path|text_orig|ipa_text``.

El texto se normaliza de forma conservadora (Unicode NFC, espacios y
separadores) antes de convertirlo con espeak-ng. Se preservan las marcas de
acento de IPA mediante ``with_stress=True``.
"""

import argparse
import csv
import re
import unicodedata
from pathlib import Path

import librosa
import soundfile as sf
from phonemizer.backend import EspeakBackend


DIALECTS = ("es_ar", "es_co", "es_pe", "es_ve", "es_cl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw_dir", default="data/openslr_raw")
    parser.add_argument("--wav_dir", default="data/wavs_22k")
    parser.add_argument("--metadata", default="data/metadata.txt")
    parser.add_argument("--sample_rate", type=int, default=22050)
    parser.add_argument(
        "--njobs",
        type=int,
        default=4,
        help="Procesos usados por phonemizer (default: 4).",
    )
    return parser.parse_args()


def normalize_text(text: str) -> str:
    """Aplica normalizacion minima sin alterar el contenido linguistico."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("|", " ")
    return re.sub(r"\s+", " ", text).strip()


def collect_entries(raw_dir: Path, wav_dir: Path, sample_rate: int):
    entries = []
    for dialect in DIALECTS:
        corpus_dir = raw_dir / dialect / f"{dialect}_female"
        tsv_path = corpus_dir / "line_index.tsv"
        wav_base = corpus_dir / "wavs"

        if not tsv_path.is_file():
            print(f"ADVERTENCIA: no encontrado: {tsv_path}")
            continue

        count = 0
        with tsv_path.open(encoding="utf-8") as file:
            for row in csv.reader(file, delimiter="\t"):
                if len(row) < 2:
                    continue

                wav_id = row[0].strip()
                text = normalize_text(row[1])
                source = wav_base / f"{wav_id}.wav"
                if not text or not source.is_file():
                    continue

                # "ar_", "co_", etc. evitan colisiones entre dialectos.
                dialect_code = dialect.split("_", maxsplit=1)[1]
                destination = wav_dir / f"{dialect_code}_{wav_id}.wav"
                if not destination.is_file():
                    audio, _ = librosa.load(source, sr=sample_rate, mono=True)
                    sf.write(destination, audio, sample_rate)

                entries.append((destination.as_posix(), text))
                count += 1

        print(f"{dialect}: {count} audios procesados")
    return entries


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir).expanduser()
    wav_dir = Path(args.wav_dir).expanduser()
    metadata = Path(args.metadata).expanduser()

    if args.sample_rate <= 0:
        raise ValueError("--sample_rate debe ser mayor que cero")
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"No existe el directorio de entrada: {raw_dir}")

    wav_dir.mkdir(parents=True, exist_ok=True)
    metadata.parent.mkdir(parents=True, exist_ok=True)

    entries = collect_entries(raw_dir, wav_dir, args.sample_rate)
    if not entries:
        raise RuntimeError("No se encontraron pares validos de audio y texto")

    print(f"Convirtiendo {len(entries)} frases a IPA...")
    backend = EspeakBackend(
        "es", with_stress=True, language_switch="remove-flags"
    )
    ipa_texts = backend.phonemize(
        [text for _, text in entries], njobs=args.njobs
    )

    with metadata.open("w", encoding="utf-8", newline="\n") as file:
        for (wav_path, text), ipa in zip(entries, ipa_texts):
            ipa_clean = re.sub(r"\s+", " ", ipa.replace("|", " ")).strip()
            file.write(f"{wav_path}|{text}|{ipa_clean}\n")

    print(f"Metadata generado: {metadata}")
    print(f"Total de entradas: {len(entries)}")


if __name__ == "__main__":
    main()
