"""
04_stratified_analysis.py
=========================
Análisis complementario por tipo de estructura prosódica.

Tipos de frases evaluados:
  - pregunta        : frases interrogativas sin pausas internas
  - pregunta+coma   : frases interrogativas con coma (pausa interna)
  - coma            : frases declarativas con coma (pausa interna)
  - declarativa     : frases declarativas simples

Motivación:
  El subconjunto histórico de 10 frases puede revelar diferencias
  importantes entre tipos de estructuras prosódicas. Un modelo puede ser
  mejor en preguntas pero peor en frases con pausas, o viceversa.
  Este análisis revela dónde cada modelo tiene ventaja o limitación.

USO:
  python stratified_analysis.py --help

Requiere:
  pip install numpy torch librosa tabulate
"""

import argparse
import os
import numpy as np
import torch
import librosa

from compute_metrics import (compute_mcd, compute_f0_metrics,
                              MAPEO, INDICES_VAL)

# ── Clasificación prosódica de las 10 frases ──────────────────────────────────
# orig_idx (posición en INDICES_VAL) → tipo prosódico
TIPOS_PROSODICOS = {
    0: 'pregunta',        # ¿Cómo se hace la soda?
    1: 'pregunta',        # ¿Cuánto cuesta esa bolsa italiana?
    2: 'pregunta',        # ¿Qué películas hay esta tarde?
    3: 'coma',            # Los requisitos para la escuela de cine...
    4: 'pregunta+coma',   # ¿Cuál es la diferencia entre obra y pieza...?
    5: 'pregunta+coma',   # ¿Qué sistema operativo recomiendas...?
    6: 'coma',            # Mañana vaya abrigada, un gorro de lana...
    7: 'coma',            # Las tejas rojas del tejado no son de México...
    8: 'declarativa',     # Me parece muy interesante el libro...
    9: 'declarativa',     # Los boletos cuestan alrededor de tres mil pesos...
}

ORDEN_TIPOS = ['pregunta', 'pregunta+coma', 'coma', 'declarativa']
# ─────────────────────────────────────────────────────────────────────────────


def compute_metrics_by_sentence(eval_dir: str, val_list: str,
                                  data_dir: str) -> dict:
    """
    Calcula MCD y F0 Corr para cada frase individualmente.

    Returns:
        dict {orig_idx: {'mcd': float, 'rmse': float, 'corr': float}}
    """
    with open(val_list) as f:
        lines = [l.strip() for l in f]

    gt_wavs = [lines[i].split('|')[0] for i in INDICES_VAL]
    gt_mels = [
        os.path.join(data_dir, 'features', 'mels',
                     os.path.basename(w).replace('.wav', '') + '.pt')
        for w in gt_wavs
    ]

    results = {}
    for audio_idx, orig_idx in sorted(MAPEO.items()):
        mel_path = os.path.join(eval_dir, f'mel_{audio_idx}.npy')
        wav_path = os.path.join(eval_dir, f'audio_{audio_idx}.wav')

        if not os.path.exists(mel_path):
            continue

        syn_mel = np.load(mel_path)
        if syn_mel.shape[0] > syn_mel.shape[1]:
            syn_mel = syn_mel.T

        gt_mel = torch.load(gt_mels[orig_idx], weights_only=True).numpy()
        mcd    = compute_mcd(gt_mel, syn_mel)
        rmse, corr = compute_f0_metrics(gt_wavs[orig_idx], wav_path)

        results[orig_idx] = {
            'mcd':  mcd,
            'rmse': rmse,
            'corr': corr,
        }

    return results


def stratified_analysis(models: dict, val_list: str, data_dir: str):
    """
    Compara múltiples modelos con análisis estratificado por tipo prosódico.

    Args:
        models : dict {nombre_modelo: eval_dir}
        val_list, data_dir : rutas del corpus
    """
    # Calcular métricas por frase para cada modelo
    all_metrics = {}
    for model_name, eval_dir in models.items():
        all_metrics[model_name] = compute_metrics_by_sentence(
            eval_dir, val_list, data_dir)

    model_names = list(models.keys())

    # ── Tabla global ──────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("ANÁLISIS ESTRATIFICADO POR TIPO DE FRASE")
    print(f"{'='*70}")

    # Encabezado
    header = f"{'Tipo':<18} {'n':<4}"
    for m in model_names:
        header += f"{'MCD '+m:>12} {'F0Corr '+m:>12}"
    print(header)
    print("-" * (22 + 24 * len(model_names)))

    # Resultados por tipo
    for tipo in ORDEN_TIPOS:
        indices_tipo = [i for i, t in TIPOS_PROSODICOS.items() if t == tipo]
        n = len(indices_tipo)

        row = f"{tipo:<18} {n:<4}"
        mcds_por_modelo  = {}
        corrs_por_modelo = {}

        for model_name in model_names:
            mcds  = [all_metrics[model_name][i]['mcd']
                     for i in indices_tipo
                     if i in all_metrics[model_name]]
            corrs = [all_metrics[model_name][i]['corr']
                     for i in indices_tipo
                     if i in all_metrics[model_name]
                     and all_metrics[model_name][i]['corr'] is not None]

            mcd_mean  = np.mean(mcds)  if mcds  else float('nan')
            corr_mean = np.mean(corrs) if corrs else float('nan')
            mcds_por_modelo[model_name]  = mcd_mean
            corrs_por_modelo[model_name] = corr_mean
            row += f"{mcd_mean:>12.2f} {corr_mean:>12.4f}"

        print(row)

        # Indicar qué modelo gana en cada métrica
        if len(model_names) == 2:
            m1, m2 = model_names
            if not np.isnan(mcds_por_modelo[m1]) and \
               not np.isnan(mcds_por_modelo[m2]):
                winner_mcd  = m1 if mcds_por_modelo[m1]  < mcds_por_modelo[m2]  else m2
                winner_corr = m1 if corrs_por_modelo[m1] > corrs_por_modelo[m2] else m2
                print(f"  {'':18} {'':4} "
                      f"{'↑ '+winner_mcd:>12} {'↑ '+winner_corr:>12}")

    # ── Resumen global ────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("PROMEDIOS GLOBALES")
    print(f"{'='*70}")
    header2 = f"{'Modelo':<25} {'MCD':>8} {'F0 RMSE':>10} {'F0 Corr':>10}"
    print(header2)
    print("-" * 55)

    for model_name in model_names:
        metrics = all_metrics[model_name]
        mcds  = [v['mcd']  for v in metrics.values()]
        rmses = [v['rmse'] for v in metrics.values() if v['rmse'] is not None]
        corrs = [v['corr'] for v in metrics.values() if v['corr'] is not None]
        print(f"{model_name:<25} {np.mean(mcds):>8.2f} "
              f"{np.mean(rmses):>10.2f} {np.mean(corrs):>10.4f}")

    # ── Análisis por frase individual ─────────────────────────────────────
    print(f"\n{'='*70}")
    print("DETALLE POR FRASE")
    print(f"{'='*70}")

    with open(val_list) as f:
        val_lines = [l.strip() for l in f]

    for orig_idx in sorted(TIPOS_PROSODICOS.keys()):
        tipo   = TIPOS_PROSODICOS[orig_idx]
        frase  = val_lines[INDICES_VAL[orig_idx]].split('|')[1][:40]
        print(f"\n  [{tipo}] {frase}")
        for model_name in model_names:
            if orig_idx not in all_metrics[model_name]:
                continue
            m = all_metrics[model_name][orig_idx]
            corr_str = f"{m['corr']:.4f}" if m['corr'] is not None else "N/A"
            print(f"    {model_name:<20} MCD={m['mcd']:.2f} "
                  f"F0Corr={corr_str}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Análisis estratificado por tipo de frase prosódica')
    parser.add_argument('--baseline-dir', required=True,
                        help='Directorio de evaluación del baseline')
    parser.add_argument('--ddpm-dir',     required=True,
                        help='Directorio de evaluación de DDPM+BETO')
    parser.add_argument('--val-list',     required=True,
                        help='Filelist de validación')
    parser.add_argument('--data-dir',     required=True,
                        help='Directorio con mels/ GT')
    args = parser.parse_args()

    models = {
        'Baseline': args.baseline_dir,
        'DDPM+BETO': args.ddpm_dir,
    }

    stratified_analysis(
        models   = models,
        val_list = args.val_list,
        data_dir = args.data_dir,
    )
