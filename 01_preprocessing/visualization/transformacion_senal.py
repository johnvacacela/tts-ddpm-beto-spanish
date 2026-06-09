#!/usr/bin/env python3
"""Genera figuras metodologicas sobre la transformacion de la senal.

Este script es auxiliar y no forma parte obligatoria del pipeline de
preprocesamiento.
"""

import argparse
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


sns.set_theme(style="ticks", palette="muted")
plt.rcParams.update(
    {
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 300,
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav_path", help="Audio WAV usado para la figura.")
    parser.add_argument(
        "--out_path", default="transformacion_senal.pdf",
        help="PDF de salida.",
    )
    parser.add_argument("--sample_rate", type=int, default=22050)
    return parser.parse_args()


def plot_signal_transformation(
    wav_path: Path, out_path: Path, sample_rate: int
) -> None:
    audio, original_sr = librosa.load(wav_path, sr=None, mono=True)
    rms = librosa.feature.rms(
        y=audio, frame_length=2048, hop_length=512
    )[0]
    rms_times = librosa.frames_to_time(
        np.arange(len(rms)), sr=original_sr, hop_length=512
    )
    threshold = np.max(rms) * 0.05
    trimmed, trim_index = librosa.effects.trim(audio, top_db=20)
    resampled = librosa.resample(
        trimmed, orig_sr=original_sr, target_sr=sample_rate
    )
    pad_start = int(0.150 * sample_rate)
    pad_end = int(0.200 * sample_rate)
    final_audio = np.pad(resampled, (pad_start, pad_end), mode="constant")

    figure, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))
    original_time = np.arange(len(audio)) / original_sr
    ax1.plot(original_time, audio, color="#7f8c8d", linewidth=0.5)
    ax1.set_title("(a) Senal original", loc="left", fontweight="bold")
    ax1.set_ylabel("Amplitud")

    ax2.plot(
        original_time, audio, color="#bdc3c7", linewidth=0.3, alpha=0.5
    )
    ax2.plot(rms_times, rms, color="#e67e22", linewidth=2, label="RMS")
    ax2.axhline(
        threshold, color="#e74c3c", linestyle="--", label="Umbral ilustrativo"
    )
    ax2.axvline(trim_index[0] / original_sr, color="#2c3e50", linestyle=":")
    ax2.axvline(trim_index[1] / original_sr, color="#2c3e50", linestyle=":")
    ax2.set_title(
        "(b) Energia RMS y limites de recorte", loc="left", fontweight="bold"
    )
    ax2.set_ylabel("Energia")
    ax2.legend(loc="upper right")

    final_time = np.arange(len(final_audio)) / sample_rate
    ax3.plot(final_time, final_audio, color="#2c3e50", linewidth=0.5)
    ax3.axvspan(0, 0.150, color="#2ecc71", alpha=0.2, label="Padding 150 ms")
    ax3.axvspan(
        final_time[-1] - 0.200,
        final_time[-1],
        color="#3498db",
        alpha=0.2,
        label="Padding 200 ms",
    )
    ax3.set_title(
        f"(c) Senal remuestreada a {sample_rate} Hz",
        loc="left",
        fontweight="bold",
    )
    ax3.set_xlabel("Tiempo (s)")
    ax3.set_ylabel("Amplitud")
    ax3.legend(loc="upper right")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(figure)
    print(f"Figura guardada en: {out_path}")


def main() -> None:
    args = parse_args()
    wav_path = Path(args.wav_path).expanduser()
    out_path = Path(args.out_path).expanduser()
    if not wav_path.is_file():
        raise FileNotFoundError(f"No existe el audio: {wav_path}")
    plot_signal_transformation(wav_path, out_path, args.sample_rate)


if __name__ == "__main__":
    main()
