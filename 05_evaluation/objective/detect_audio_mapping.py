"""
Utilidad auxiliar para detectar el reordenamiento de FastPitch en inferencia.

PROBLEMA:
  FastPitch ordena las frases por longitud de tokens para optimizar
  el procesamiento en batch. El audio_0.wav NO corresponde a la línea 0
  del archivo de entrada — corresponde a la frase más larga.

SOLUCIÓN:
  Ejecutar inferencia con bs=1 (sin reordenamiento posible) y comparar
  el log de salida con el archivo de entrada para construir el mapeo.

USO:
  1. Ejecutar inferencia con --save-mels y capturar stdout a un log:
       python3 inference.py -i frases.txt -o eval_dir ... 2>&1 | tee inf.log

  2. Ejecutar este script:
       python 01_detect_audio_mapping.py --input frases.txt --log inf.log

  El mapeo fijo de 10 frases se conserva solo como ejemplo histórico y no
  representa la evaluación final de la tesis.

MAPEO AUXILIAR para el subconjunto histórico de 10 frases:
  mapeo = {0:3, 1:5, 2:6, 3:8, 4:4, 5:7, 6:9, 7:1, 8:2, 9:0}
  # audio_idx → posición en el archivo de entrada (0-indexado)
"""

import argparse
import os


# Mapeo pre-calculado del ejemplo auxiliar de 10 frases.
MAPEO_ESTANDAR = {0: 3, 1: 5, 2: 6, 3: 8, 4: 4, 5: 7, 6: 9, 7: 1, 8: 2, 9: 0}

# Índices en el val set de las 10 frases de evaluación estándar
INDICES_VAL = [15, 21, 25, 36, 47, 54, 64, 84, 1, 3]


def detect_mapping_from_log(input_file: str, log_file: str) -> dict:
    """
    Detecta el mapeo audio_idx -> línea_original comparando
    el log de inferencia con el archivo de entrada.

    El log de FastPitch imprime cada frase procesada en orden de salida,
    justo antes de las métricas DLL. Estas líneas comienzan con
    caracteres IPA o letras del español.
    """
    with open(input_file, encoding='utf-8') as f:
        frases_orig = [l.strip() for l in f if l.strip()]

    # Extraer orden de salida del log
    orden_salida = []
    with open(log_file, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # Líneas que empiezan con IPA o letras (no DLL, no warnings)
            if (line and
                not line.startswith('DLL') and
                not line.startswith('/') and
                not line.startswith('Warning') and
                not line.startswith('Some') and
                not line.startswith('HiFi') and
                not line.startswith('Fast') and
                not line.startswith('You')):
                orden_salida.append(line)

    print(f"Frases en archivo de entrada: {len(frases_orig)}")
    print(f"Frases detectadas en log:     {len(orden_salida)}")

    # Construir mapeo por prefijo de texto
    mapeo = {}
    for audio_idx, frase_salida in enumerate(orden_salida):
        for orig_idx, frase_orig in enumerate(frases_orig):
            if frase_salida[:40].strip() == frase_orig[:40].strip():
                mapeo[audio_idx] = orig_idx
                break

    n_mapped = len(mapeo)
    n_total  = len(orden_salida)
    print(f"Mapeados: {n_mapped}/{n_total}")

    if n_mapped < n_total:
        sin_mapeo = [i for i in range(n_total) if i not in mapeo]
        print(f"⚠️  Sin mapeo: audio_idx {sin_mapeo}")
    else:
        print("✅ Todos los audios mapeados correctamente")

    return mapeo


def verify_mapping(mapeo: dict, eval_dir: str,
                   val_list: str, indices_val: list,
                   data_dir: str):
    """
    Verifica el mapeo calculando MCD para un par de frases
    y comparando con el valor esperado.
    """
    import numpy as np
    import torch

    with open(val_list) as f:
        lines = [l.strip() for l in f]

    gt_mels = []
    for i in indices_val:
        wav = lines[i].split('|')[0]
        name = os.path.basename(wav).replace('.wav', '')
        gt_mels.append(os.path.join(
            data_dir, 'features', 'mels', f'{name}.pt'))

    print("\nVerificando mapeo con MCD:")
    for audio_idx, orig_idx in sorted(mapeo.items())[:3]:
        mel_path = f'{eval_dir}/mel_{audio_idx}.npy'
        if not os.path.exists(mel_path):
            continue
        syn = np.load(mel_path)
        if syn.shape[0] > syn.shape[1]:
            syn = syn.T
        gt = torch.load(gt_mels[orig_idx], weights_only=True).numpy()
        T   = min(gt.shape[1], syn.shape[1])
        mcd = np.mean(np.sqrt(2 * np.sum((gt[:, :T] - syn[:, :T])**2, axis=0)))
        print(f"  audio_{audio_idx} → orig_{orig_idx}: MCD={mcd:.2f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detecta mapeo de audios de FastPitch')
    parser.add_argument('--input', type=str, help='Archivo de frases de entrada')
    parser.add_argument('--log',   type=str, help='Log de inferencia de FastPitch')
    parser.add_argument('--use-standard', action='store_true',
                        help='Usar el mapeo histórico auxiliar de 10 frases')
    args = parser.parse_args()

    if args.use_standard:
        print("Usando mapeo histórico auxiliar:")
        print(f"mapeo = {MAPEO_ESTANDAR}")
    elif args.input and args.log:
        mapeo = detect_mapping_from_log(args.input, args.log)
        print(f"\nmapeo = {mapeo}")
    else:
        print("Uso:")
        print("  python detect_audio_mapping.py --input frases.txt --log inf.log")
        print("  python detect_audio_mapping.py --use-standard")
        print(f"\nMapeo auxiliar (10 frases): {MAPEO_ESTANDAR}")
