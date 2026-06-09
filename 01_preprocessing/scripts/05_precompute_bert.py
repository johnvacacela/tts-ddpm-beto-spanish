#!/usr/bin/env python3
"""Precomputa embeddings semanticos BETO a partir del vector [CLS].

El Camino semantico (WordPiece) usa por defecto
``dccuchile/bert-base-spanish-wwm-cased`` y guarda un tensor de 768
dimensiones por frase.
"""

import argparse
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_file", default="filelists/slr_bert_train.txt")
    parser.add_argument("--val_file", default="filelists/slr_bert_val.txt")
    parser.add_argument("--out_dir", default="data/bert_embeddings")
    parser.add_argument(
        "--model_name",
        default="dccuchile/bert-base-spanish-wwm-cased",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    return parser.parse_args()


def read_records(filelist: Path):
    records = []
    with filelist.open(encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) < 3:
                raise ValueError(
                    f"Formato invalido en {filelist}:{line_number}"
                )
            wav_path = Path(parts[0]).expanduser()
            beto_text = parts[3] if len(parts) >= 4 else parts[1]
            records.append((wav_path.stem, beto_text))
    return records


def main() -> None:
    args = parse_args()
    train_file = Path(args.train_file).expanduser()
    val_file = Path(args.val_file).expanduser()
    out_dir = Path(args.out_dir).expanduser()

    for path in (train_file, val_file):
        if not path.is_file():
            raise FileNotFoundError(f"No existe el filelist: {path}")
    if args.batch_size <= 0:
        raise ValueError("--batch_size debe ser mayor que cero")

    out_dir.mkdir(parents=True, exist_ok=True)
    records = list(dict.fromkeys(
        read_records(train_file) + read_records(val_file)
    ))
    pending = [
        record for record in records
        if not (out_dir / f"{record[0]}.pt").is_file()
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Cargando {args.model_name} en {device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(device)
    model.eval()

    for start in range(0, len(pending), args.batch_size):
        batch = pending[start:start + args.batch_size]
        names, texts = zip(*batch)
        tokens = tokenizer(
            list(texts),
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(device)
        with torch.inference_mode():
            cls_vectors = model(**tokens).last_hidden_state[:, 0, :].cpu()
        for name, vector in zip(names, cls_vectors):
            torch.save(vector.float(), out_dir / f"{name}.pt")
        print(f"Progreso: {min(start + len(batch), len(pending))}/{len(pending)}")

    print(f"Embeddings disponibles: {len(list(out_dir.glob('*.pt')))}")
    print(f"Directorio: {out_dir}")


if __name__ == "__main__":
    main()
