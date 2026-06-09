"""
Calcula MCD, F0 RMSE y F0 Corr para evaluación objetiva TTS.

Puede utilizarse sobre el test independiente o sobre validación. Las
constantes de 10 frases se conservan únicamente para reproducir el análisis
auxiliar histórico; no definen la evaluación final de la tesis.

Métricas:
  MCD  (Mel Cepstral Distortion) — distancia espectral. Menor es mejor.
  F0 RMSE — error cuadrático medio del pitch en Hz
         Menor es mejor.
  F0 Corr — correlación de Pearson del contorno de pitch
         Mayor es mejor. Rango [-1, 1].

USO:
  python compute_metrics.py --eval-dir ... --val-list ... --data-dir ...

Requiere:
  pip install numpy torch librosa
"""

import argparse
import os
import numpy as np
import torch
import librosa

# ── Mapeo y frases de evaluación estándar ─────────────────────────────────────
MAPEO       = {0: 3, 1: 5, 2: 6, 3: 8, 4: 4, 5: 7, 6: 9, 7: 1, 8: 2, 9: 0}
INDICES_VAL = [15, 21, 25, 36, 47, 54, 64, 84, 1, 3]
TIPOS       = {
    0: 'pregunta',      1: 'pregunta',      2: 'pregunta',
    3: 'coma',          4: 'pregunta+coma', 5: 'pregunta+coma',
    6: 'coma',          7: 'coma',          8: 'declarativa',
    9: 'declarativa'
}
# ─────────────────────────────────────────────────────────────────────────────


def compute_mcd(mel_gt: np.ndarray, mel_syn: np.ndarray) -> float:
    """
    Mel Cepstral Distortion entre mel GT y mel sintetizado.
    Alinea por el mínimo de frames antes de calcular.
    """
    T    = min(mel_gt.shape[1], mel_syn.shape[1])
    diff = mel_gt[:, :T] - mel_syn[:, :T]
    return float(np.mean(np.sqrt(2 * np.sum(diff ** 2, axis=0))))


def compute_f0_metrics(wav_gt: str, wav_syn: str, sr: int = 22050,
                       hop: int = 256, fmin: float = 65.0,
                       fmax: float = 500.0):
    """
    F0 RMSE y F0 Correlation usando pyin de librosa.
    Solo considera regiones donde ambas señales tienen pitch (voiced).

    Returns:
        rmse : float o None si no hay suficientes frames voiced
        corr : float o None
    """
    y_gt,  _ = librosa.load(wav_gt,  sr=sr)
    y_syn, _ = librosa.load(wav_syn, sr=sr)

    f0_gt, _, _ = librosa.pyin(
        y_gt, sr=sr, hop_length=hop, fmin=fmin, fmax=fmax)
    f0_syn, _, _ = librosa.pyin(
        y_syn, sr=sr, hop_length=hop, fmin=fmin, fmax=fmax)

    T      = min(len(f0_gt), len(f0_syn))
    f0_gt  = f0_gt[:T]
    f0_syn = f0_syn[:T]
    voiced = (~np.isnan(f0_gt)) & (~np.isnan(f0_syn)) & (f0_gt > 0) & (f0_syn > 0)

    if voiced.sum() < 2:
        return None, None

    rmse = float(np.sqrt(np.mean((f0_gt[voiced] - f0_syn[voiced]) ** 2)))
    corr = float(np.corrcoef(f0_gt[voiced], f0_syn[voiced])[0, 1])
    return rmse, corr


def evaluate(eval_dir: str, val_list: str, data_dir: str,
             model_name: str, mapeo: dict = None,
             indices_val: list = None, pitch_fmin: float = 65.0,
             pitch_fmax: float = 500.0):
    """
    Evalúa un directorio de audios sintetizados por FastPitch.

    Args:
        eval_dir    : directorio con audio_N.wav y mel_N.npy
        val_list    : filelist de validación de FastPitch
        data_dir    : directorio con mels/ GT
        model_name  : nombre del modelo para el reporte
        mapeo       : dict {audio_idx: orig_idx} — ver 01_detect_audio_mapping.py
        indices_val : índices en val_list de las frases de evaluación
    """
    if mapeo is None:
        mapeo = MAPEO
    if indices_val is None:
        indices_val = INDICES_VAL

    with open(val_list) as f:
        lines = [l.strip() for l in f]

    gt_wavs = [lines[i].split('|')[0] for i in indices_val]
    gt_mels = [
        os.path.join(data_dir, 'features', 'mels',
                     os.path.basename(w).replace('.wav', '') + '.pt')
        for w in gt_wavs
    ]

    print(f"\n{'='*70}")
    print(f"EVALUACIÓN — {model_name}")
    print(f"{'='*70}")
    print(f"{'#':<4} {'Tipo':<15} {'MCD':>7} {'F0 RMSE':>9} {'F0 Corr':>9}")
    print("-" * 50)

    mcds, rmses, corrs = [], [], []
    por_tipo = {}

    for audio_idx, orig_idx in sorted(mapeo.items()):
        mel_path = os.path.join(eval_dir, f'mel_{audio_idx}.npy')
        wav_path = os.path.join(eval_dir, f'audio_{audio_idx}.wav')

        if not os.path.exists(mel_path) or not os.path.exists(wav_path):
            print(f"⚠️  Falta: {mel_path}")
            continue

        syn_mel = np.load(mel_path)
        if syn_mel.shape[0] > syn_mel.shape[1]:
            syn_mel = syn_mel.T

        gt_mel = torch.load(gt_mels[orig_idx], weights_only=True).numpy()
        mcd    = compute_mcd(gt_mel, syn_mel)
        rmse, corr = compute_f0_metrics(
            gt_wavs[orig_idx], wav_path,
            fmin=pitch_fmin, fmax=pitch_fmax)

        tipo = TIPOS.get(orig_idx, 'declarativa')
        mcds.append(mcd)

        if tipo not in por_tipo:
            por_tipo[tipo] = {'mcds': [], 'rmses': [], 'corrs': []}
        por_tipo[tipo]['mcds'].append(mcd)

        if rmse is not None:
            rmses.append(rmse)
            corrs.append(corr)
            por_tipo[tipo]['rmses'].append(rmse)
            por_tipo[tipo]['corrs'].append(corr)
            print(f"{orig_idx:<4} {tipo:<15} {mcd:>7.2f} {rmse:>9.2f} {corr:>9.4f}")

    print("-" * 50)
    print(f"{'PROMEDIO':<19} {np.mean(mcds):>7.2f} "
          f"{np.mean(rmses):>9.2f} {np.mean(corrs):>9.4f}")

    print(f"\n--- Análisis estratificado ---")
    for tipo in ['pregunta', 'pregunta+coma', 'coma', 'declarativa']:
        if tipo not in por_tipo:
            continue
        r = por_tipo[tipo]
        print(f"  {tipo:<20} n={len(r['mcds'])} "
              f"MCD={np.mean(r['mcds']):.2f} "
              f"F0Corr={np.mean(r['corrs']):.4f}")

    return {
        'mcd':   np.mean(mcds),
        'rmse':  np.mean(rmses),
        'corr':  np.mean(corrs),
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calcula métricas objetivas TTS')
    parser.add_argument('--eval-dir',   required=True,
                        help='Directorio con audios sintetizados')
    parser.add_argument('--val-list',   required=True,
                        help='Filelist de validación')
    parser.add_argument('--data-dir',   required=True,
                        help='Directorio con mels/ GT')
    parser.add_argument('--model-name', default='Modelo',
                        help='Nombre del modelo para el reporte')
    parser.add_argument('--pitch-fmin', type=float, default=65.0)
    parser.add_argument('--pitch-fmax', type=float, default=500.0)
    args = parser.parse_args()

    evaluate(
        eval_dir   = args.eval_dir,
        val_list   = args.val_list,
        data_dir   = args.data_dir,
        model_name = args.model_name,
        pitch_fmin = args.pitch_fmin,
        pitch_fmax = args.pitch_fmax,
    )
