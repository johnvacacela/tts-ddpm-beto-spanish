#!/usr/bin/env python3
"""Precomputa caracteristicas acusticas del corpus.

El pitch/F0 se precomputa como caracteristica usada por los modelos. La
energia RMS es auxiliar para analisis, control y validacion; no es un predictor
prosodico explicito de la arquitectura final.

Los Mel-espectrogramas pueden calcularse dinamicamente durante el entrenamiento
segun la implementacion de FastPitch. Los Mels generados aqui son auxiliares
para inspeccion del corpus, validacion del preprocesamiento y evaluacion
espectral objetiva, por ejemplo MCD.
"""

import argparse
from pathlib import Path

import librosa
import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_file", default="filelists/slr_all_train.txt")
    parser.add_argument("--val_file", default="filelists/slr_all_val.txt")
    parser.add_argument("--out_dir", default="data/features")
    parser.add_argument("--sample_rate", type=int, default=22050)
    parser.add_argument("--n_fft", type=int, default=1024)
    parser.add_argument("--hop_length", type=int, default=256)
    parser.add_argument("--win_length", type=int, default=1024)
    parser.add_argument("--n_mels", type=int, default=80)
    parser.add_argument("--fmin", type=float, default=0)
    parser.add_argument("--fmax", type=float, default=8000)
    parser.add_argument("--pitch_fmin", type=float, default=65)
    parser.add_argument("--pitch_fmax", type=float, default=500)
    return parser.parse_args()


def read_wav_paths(filelist: Path):
    paths = []
    with filelist.open(encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line:
                continue
            wav_path = line.split("|", maxsplit=1)[0]
            if not wav_path:
                raise ValueError(f"Ruta vacia en {filelist}:{line_number}")
            paths.append(Path(wav_path).expanduser())
    return paths


def extract_features(wav_path: Path, output_dirs, args) -> None:
    audio, _ = librosa.load(wav_path, sr=args.sample_rate, mono=True)
    sample_id = wav_path.stem

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=args.sample_rate,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length,
        n_mels=args.n_mels,
        fmin=args.fmin,
        fmax=args.fmax,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    f0, voiced_flag, voiced_prob = librosa.pyin(
        audio,
        fmin=args.pitch_fmin,
        fmax=args.pitch_fmax,
        sr=args.sample_rate,
        frame_length=2048,
        hop_length=args.hop_length,
    )
    del voiced_flag, voiced_prob
    # Los ceros representan regiones no sonoras/no vocalizadas.
    f0 = np.nan_to_num(f0, nan=0.0)

    rms = librosa.feature.rms(
        y=audio, frame_length=2048, hop_length=args.hop_length
    )[0]

    torch.save(
        torch.from_numpy(mel_db).float(),
        output_dirs["mels"] / f"{sample_id}.pt",
    )
    torch.save(
        torch.from_numpy(f0).float(),
        output_dirs["pitch"] / f"{sample_id}.pt",
    )
    torch.save(
        torch.from_numpy(rms).float(),
        output_dirs["energy"] / f"{sample_id}.pt",
    )


def main() -> None:
    args = parse_args()
    train_file = Path(args.train_file).expanduser()
    val_file = Path(args.val_file).expanduser()
    out_dir = Path(args.out_dir).expanduser()

    for path in (train_file, val_file):
        if not path.is_file():
            raise FileNotFoundError(f"No existe el filelist: {path}")
    if not 0 <= args.fmin < args.fmax <= args.sample_rate / 2:
        raise ValueError("El rango Mel debe respetar 0 <= fmin < fmax <= Nyquist")
    if not 0 < args.pitch_fmin < args.pitch_fmax <= args.sample_rate / 2:
        raise ValueError("El rango de pitch debe ser positivo y menor que Nyquist")

    output_dirs = {
        name: out_dir / name for name in ("mels", "pitch", "energy")
    }
    for directory in output_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    wav_paths = list(dict.fromkeys(
        read_wav_paths(train_file) + read_wav_paths(val_file)
    ))
    completed = 0
    errors = 0
    for index, wav_path in enumerate(wav_paths, start=1):
        sample_id = wav_path.stem
        expected = [
            directory / f"{sample_id}.pt"
            for directory in output_dirs.values()
        ]
        if all(path.is_file() for path in expected):
            completed += 1
            continue
        try:
            extract_features(wav_path, output_dirs, args)
            completed += 1
        except Exception as error:
            errors += 1
            print(f"ERROR en {wav_path}: {error}")
        if index % 500 == 0 or index == len(wav_paths):
            print(f"Progreso: {index}/{len(wav_paths)}")

    print(f"Completado: {completed} OK, {errors} errores")
    for name, directory in output_dirs.items():
        print(f"{name}: {len(list(directory.glob('*.pt')))} archivos")


if __name__ == "__main__":
    main()
