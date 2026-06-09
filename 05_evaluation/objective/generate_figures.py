"""
05_generate_figures.py
======================
Genera figuras auxiliares entre GT, Baseline FastPitch y DDPM+BETO.

Figuras generadas:
  spec_{frase}.png   Espectrograma GT vs Baseline vs DDPM+BETO
  pitch_{frase}.png  Contorno de pitch GT vs Baseline vs DDPM+BETO

USO:
  python generate_figures.py --help

Requiere:
  pip install matplotlib librosa numpy torch
"""

import argparse
import os
import numpy as np
import torch
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from compute_metrics import MAPEO, INDICES_VAL

# ── Frases seleccionadas para figuras ─────────────────────────────────────────
# orig_idx: label para el título
FRASES_FIGURA = {
    0:  "¿Cómo se hace la soda?",
    5:  "¿Qué sistema operativo recomiendas?",
    3:  "Los requisitos para la escuela de cine",
}
# audio_idx correspondiente para cada orig_idx
AUDIO_IDX_MAP = {v: k for k, v in MAPEO.items()}

SR  = 22050
HOP = 256


def load_mel_syn(eval_dir: str, audio_idx: int) -> np.ndarray:
    """Carga un mel sintetizado y lo transpone si es necesario."""
    mel = np.load(os.path.join(eval_dir, f'mel_{audio_idx}.npy'))
    if mel.shape[0] > mel.shape[1]:
        mel = mel.T
    return mel


def plot_spectrogram(orig_idx: int, label: str,
                     eval_dirs: dict,
                     gt_mels: list, audio_idx: int,
                     output_dir: str):
    """Genera figura de espectrogramas comparativos."""
    n_models = len(eval_dirs)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
    if n_models == 1:
        axes = [axes]

    fig.suptitle(f'Espectrograma Mel — "{label}"', fontsize=13, y=1.02)

    for ax, (model_name, eval_dir) in zip(axes, eval_dirs.items()):
        if model_name == 'GT':
            mel = torch.load(gt_mels[orig_idx], weights_only=True).numpy()
        else:
            mel = load_mel_syn(eval_dir, audio_idx)

        im = ax.imshow(mel, aspect='auto', origin='lower',
                       cmap='magma', interpolation='nearest',
                       vmin=-10, vmax=0)
        ax.set_title(model_name, fontsize=12, fontweight='bold')
        ax.set_xlabel('Frames', fontsize=10)
        ax.set_ylabel('Mel bins', fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    safe = label.replace(' ', '_').replace('?', '').replace('.', '')[:25]
    out  = os.path.join(output_dir, f'spec_{safe}.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ {out}")


def plot_pitch(orig_idx: int, label: str,
               eval_dirs: dict,
               gt_wavs: list, audio_idx: int,
               output_dir: str):
    """Genera figura de contornos de pitch comparativos."""
    colors = {
        'GT':       '#1A237E',
        'Baseline': '#E65100',
        'DDPM+BETO': '#2E7D32',
    }

    fig, ax = plt.subplots(figsize=(12, 4))

    for model_name, eval_dir in eval_dirs.items():
        if model_name == 'GT':
            wav_path = gt_wavs[orig_idx]
        else:
            wav_path = os.path.join(eval_dir, f'audio_{audio_idx}.wav')

        if not os.path.exists(wav_path):
            continue

        y, _ = librosa.load(wav_path, sr=SR)
        f0, _, _ = librosa.pyin(
            y,
            fmin=65,
            fmax=500,
            sr=SR, hop_length=HOP)
        times  = librosa.frames_to_time(np.arange(len(f0)), sr=SR,
                                         hop_length=HOP)
        voiced = ~np.isnan(f0) & (f0 > 0)

        color = colors.get(model_name, '#000000')
        lw    = 2.5 if model_name == 'GT' else 1.8
        ls    = '-' if model_name == 'GT' else '--'
        ax.plot(times[voiced], f0[voiced],
                color=color, linewidth=lw, linestyle=ls,
                label=model_name, alpha=0.9)

    ax.set_xlabel('Tiempo (s)', fontsize=12)
    ax.set_ylabel('F0 (Hz)', fontsize=12)
    ax.set_title(f'Contorno de Pitch — "{label}"', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(50, 420)
    plt.tight_layout()

    safe = label.replace(' ', '_').replace('?', '').replace('.', '')[:25]
    out  = os.path.join(output_dir, f'pitch_{safe}.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ {out}")


def generate_all_figures(baseline_dir: str, ddpm_dir: str,
                          val_list: str, data_dir: str,
                          output_dir: str):
    """Genera todas las figuras de evaluación."""
    os.makedirs(output_dir, exist_ok=True)

    with open(val_list) as f:
        lines = [l.strip() for l in f]

    gt_wavs = [lines[i].split('|')[0] for i in INDICES_VAL]
    gt_mels = [
        os.path.join(data_dir, 'features', 'mels',
                     os.path.basename(w).replace('.wav', '') + '.pt')
        for w in gt_wavs
    ]

    # Construir diccionario de directorios de evaluación
    eval_dirs = {
        'GT': None,
        'Baseline': baseline_dir,
        'DDPM+BETO': ddpm_dir,
    }

    for orig_idx, label in FRASES_FIGURA.items():
        audio_idx = AUDIO_IDX_MAP.get(orig_idx)
        if audio_idx is None:
            continue
        plot_spectrogram(orig_idx, label, eval_dirs,
                         gt_mels, audio_idx, output_dir)
        plot_pitch(orig_idx, label, eval_dirs,
                   gt_wavs, audio_idx, output_dir)

    print(f"\n✅ Figuras generadas en {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Genera figuras de evaluación TTS')
    parser.add_argument('--baseline-dir', required=True)
    parser.add_argument('--ddpm-dir',     required=True)
    parser.add_argument('--val-list',     required=True)
    parser.add_argument('--data-dir',     required=True)
    parser.add_argument('--output-dir',   required=True)
    args = parser.parse_args()

    generate_all_figures(
        baseline_dir = args.baseline_dir,
        ddpm_dir     = args.ddpm_dir,
        val_list     = args.val_list,
        data_dir     = args.data_dir,
        output_dir   = args.output_dir,
    )
