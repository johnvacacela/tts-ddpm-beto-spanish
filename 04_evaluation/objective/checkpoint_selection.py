"""
03_checkpoint_selection.py
==========================
Utilidad auxiliar para evaluar checkpoints mediante métricas objetivas.

Metodología:
  La selección oficial de la tesis se realizó sobre el conjunto de validación
  de 1,216 frases. Los CSV oficiales se encuentran en ``results/``.
  El punto operativo se selecciona por balance entre MCD, F0 RMSE y F0 Corr.

USO:
  python checkpoint_selection.py --help

Requiere:
  pip install numpy torch librosa
"""

import argparse
import os
import subprocess
import shutil
import numpy as np
import torch
import librosa

from compute_metrics import compute_mcd, compute_f0_metrics, MAPEO, INDICES_VAL

# ─────────────────────────────────────────────────────────────────────────────
def run_inference(ckpt_path: str, fp_dir: str, out_dir: str,
                  input_file: str, hifigan: str,
                  hifigan_config: str) -> bool:
    """Ejecuta inferencia con un checkpoint específico."""
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        'python3', 'inference.py', '-i', input_file, '-o', out_dir,
        '--fastpitch', ckpt_path, '--hifigan', hifigan,
        '--hifigan-config', hifigan_config, '--symbol-set', 'ipa_es',
        '--text-cleaners', 'ipa_cleaners', '--p-arpabet', '0.0',
        '--affinity', 'disabled', '--fade-out', '0', '--cuda', '--save-mels',
    ]
    result = subprocess.run(cmd, cwd=fp_dir, capture_output=True)
    return result.returncode == 0


def eval_checkpoint(ckpt_path: str, fp_dir: str,
                    val_list: str, data_dir: str, input_file: str,
                    hifigan: str, hifigan_config: str,
                    work_dir: str) -> dict:
    """Evalúa un checkpoint y retorna las métricas."""
    out_dir = os.path.join(work_dir, 'ckpt_eval_tmp')

    if not run_inference(
            ckpt_path, fp_dir, out_dir, input_file,
            hifigan, hifigan_config):
        return None

    with open(val_list) as f:
        lines = [l.strip() for l in f]

    gt_wavs = [lines[i].split('|')[0] for i in INDICES_VAL]
    gt_mels = [
        os.path.join(data_dir, 'features', 'mels',
                     os.path.basename(w).replace('.wav', '') + '.pt')
        for w in gt_wavs
    ]

    mcds, rmses, corrs = [], [], []
    for audio_idx, orig_idx in sorted(MAPEO.items()):
        mel_path = f'{out_dir}/mel_{audio_idx}.npy'
        wav_path = f'{out_dir}/audio_{audio_idx}.wav'
        if not os.path.exists(mel_path):
            continue
        syn_mel = np.load(mel_path)
        if syn_mel.shape[0] > syn_mel.shape[1]:
            syn_mel = syn_mel.T
        gt_mel = torch.load(gt_mels[orig_idx], weights_only=True).numpy()
        mcds.append(compute_mcd(gt_mel, syn_mel))
        rmse, corr = compute_f0_metrics(gt_wavs[orig_idx], wav_path)
        if rmse is not None:
            rmses.append(rmse)
            corrs.append(corr)

    shutil.rmtree(out_dir, ignore_errors=True)

    return {
        'mcd':  np.mean(mcds)  if mcds  else None,
        'rmse': np.mean(rmses) if rmses else None,
        'corr': np.mean(corrs) if corrs else None,
    }


def select_best_checkpoint(ckpt_dir: str, fp_dir: str,
                            val_list: str, data_dir: str,
                            input_file: str, hifigan: str,
                            hifigan_config: str, work_dir: str,
                            model_name: str = 'modelo'):
    """
    Evalúa todos los checkpoints y muestra tabla comparativa.
    """
    # Encontrar checkpoints disponibles
    ckpts = sorted([
        f for f in os.listdir(ckpt_dir)
        if f.startswith('FastPitch_checkpoint_') and f.endswith('.pt')
    ], key=lambda x: int(x.split('_')[-1].replace('.pt', '')))

    if not ckpts:
        print(f"❌ No se encontraron checkpoints en {ckpt_dir}")
        return

    print(f"\n{'='*65}")
    print(f"SELECCIÓN DE CHECKPOINT — {model_name}")
    print(f"{'='*65}")
    print(f"{'Época':<8} {'MCD':>8} {'F0 RMSE':>10} {'F0 Corr':>10}")
    print("-" * 42)

    best = {'epoch': None, 'mcd': 999, 'corr': -1, 'path': None}
    results = []

    for ckpt_file in ckpts:
        epoch     = int(ckpt_file.split('_')[-1].replace('.pt', ''))
        ckpt_path = os.path.join(ckpt_dir, ckpt_file)
        metrics = eval_checkpoint(
            ckpt_path, fp_dir, val_list, data_dir, input_file,
            hifigan, hifigan_config, work_dir)

        if metrics is None or metrics['mcd'] is None:
            print(f"{epoch:<8} {'ERROR':>8}")
            continue

        results.append((epoch, metrics))
        print(f"{epoch:<8} {metrics['mcd']:>8.2f} "
              f"{metrics['rmse']:>10.2f} {metrics['corr']:>10.4f}")

    if results:
        mcd = np.array([item[1]['mcd'] for item in results])
        rmse = np.array([item[1]['rmse'] for item in results])
        corr = np.array([item[1]['corr'] for item in results])
        normalize = lambda values: (
            (values - values.min()) / (np.ptp(values) or 1.0))
        scores = normalize(mcd) + normalize(rmse) + normalize(-corr)
        best_idx = int(np.argmin(scores))
        epoch, metrics = results[best_idx]
        best = {
            'epoch': epoch, 'mcd': metrics['mcd'],
            'rmse': metrics['rmse'], 'corr': metrics['corr'],
            'path': os.path.join(
                ckpt_dir, f'FastPitch_checkpoint_{epoch}.pt'),
        }
    else:
        print("\nNo se obtuvieron métricas válidas para ningún checkpoint.")
        return None

    print("-" * 42)
    print(f"\n✅ Mejor checkpoint: época {best['epoch']}")
    print(f"   MCD={best['mcd']:.2f} | F0 RMSE={best['rmse']:.2f} "
          f"| F0 Corr={best['corr']:.4f}")
    print(f"   Path: {best['path']}")

    return best


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Selecciona el checkpoint óptimo por métricas objetivas')
    parser.add_argument('--ckpt-dir',  required=True,
                        help='Directorio con checkpoints .pt')
    parser.add_argument('--fp-dir',    required=True,
                        help='Directorio con inference.py de FastPitch')
    parser.add_argument('--val-list',  required=True,
                        help='Filelist de validación')
    parser.add_argument('--data-dir',  required=True,
                        help='Directorio con mels/ GT')
    parser.add_argument('--model',     default='modelo',
                        help='Nombre del modelo')
    parser.add_argument('--input-file', required=True,
                        help='Frases usadas para la inferencia auxiliar')
    parser.add_argument('--hifigan', required=True)
    parser.add_argument('--hifigan-config', required=True)
    parser.add_argument('--work-dir', default='eval/checkpoint_selection')
    args = parser.parse_args()

    select_best_checkpoint(
        ckpt_dir   = args.ckpt_dir,
        fp_dir     = args.fp_dir,
        val_list   = args.val_list,
        data_dir   = args.data_dir,
        input_file = args.input_file,
        hifigan = args.hifigan,
        hifigan_config = args.hifigan_config,
        work_dir = args.work_dir,
        model_name = args.model,
    )
