# Modelo 1: Baseline FastPitch adaptado al español

Esta carpeta documenta el **Baseline FastPitch** de la tesis, adaptado para
entrenar con transcripciones IPA en español latinoamericano.

El modelo conserva la arquitectura FastPitch original y su predictor
determinista de pitch. Esta carpeta **no incluye DDPM ni BETO**; esas
extensiones corresponden al modelo propuesto y se mantienen separadas.

La implementación parte del repositorio:

- [NVIDIA DeepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples)
- Ruta base:
  `DeepLearningExamples/PyTorch/SpeechSynthesis/FastPitch`

Solo se versionan los archivos modificados respecto a la implementación
original de NVIDIA.

## Archivos modificados

Esta carpeta contiene exactamente cuatro archivos modificados:

```text
common/text/
├── symbols.py          Añade el symbol set `ipa_es` para transcripciones IPA en español.
├── cleaners.py         Añade `ipa_cleaners` para preservar símbolos IPA y normalizar espacios.
└── text_processing.py  Registra `ipa_es` en el dispatcher de procesamiento textual.

prepare_dataset.py      Añade `ipa_es` como valor válido de `--symbol_set`.
```

Para usar estos cambios, los archivos deben conservar esas rutas dentro de una
copia del directorio FastPitch original.

## Procesamiento textual

El pipeline reemplaza la representación inglesa basada en CMU Arpabet por
transcripciones IPA generadas previamente:

```text
Texto en español
→ espeak-ng / phonemizer
→ transcripción IPA con marcas de acento
→ ipa_cleaners
→ ipa_es
→ FastPitch
```

### Symbol set `ipa_es`

El conjunto `ipa_es` contiene **47 símbolos totales**, incluyendo símbolos
IPA, marcas de acento, puntuación y tokens de control:

- 1 token de padding: `_`;
- 1 separador especial: `-`;
- 9 símbolos de puntuación y espacio;
- 19 símbolos IPA base;
- 17 símbolos IPA Unicode, incluidas las marcas de acento primario `ˈ` y
  secundario `ˌ`.

No son 47 fonemas. Los diptongos no se codifican como tokens independientes;
se representan como secuencias de símbolos IPA.

El padding ocupa el índice `0`, como requiere el procesamiento original de
FastPitch.

### Cleaner `ipa_cleaners`

`ipa_cleaners` solo normaliza secuencias de espacios. No translitera, no
convierte a ASCII, no elimina símbolos IPA y no elimina las marcas de acento.
La entrada debe llegar previamente fonemizada.

## Preparación

Clone el repositorio base y copie los cuatro archivos modificados en las rutas
equivalentes de FastPitch:

```bash
git clone https://github.com/NVIDIA/DeepLearningExamples.git

export PROJECT_ROOT=/ruta/al/proyecto
export FASTPITCH_ROOT=$PROJECT_ROOT/external/FastPitch
export DATA_DIR=$PROJECT_ROOT/data
export OUT_DIR=$PROJECT_ROOT/outputs/baseline_fastpitch
export HIFIGAN_PATH=$PROJECT_ROOT/external/hifigan/g_02500000
export HIFIGAN_CONFIG=$PROJECT_ROOT/external/hifigan/config.json
```

`FASTPITCH_ROOT` debe apuntar a una copia funcional de:

```text
DeepLearningExamples/PyTorch/SpeechSynthesis/FastPitch
```

Los filelists del baseline usan el formato:

```text
wav_path|text_orig|ipa_text
```

## Entrenamiento

El entrenamiento final no utiliza condicionamiento por energía. El Pitch/F0
se carga desde disco y FastPitch conserva su predictor de pitch original.

```bash
cd "$FASTPITCH_ROOT"

mkdir -p "$OUT_DIR/logs"

torchrun --nproc_per_node=2 train.py \
    --cuda --amp \
    -o "$OUT_DIR" \
    --log-file "$OUT_DIR/logs/train.log" \
    --dataset-path "$DATA_DIR" \
    --training-files "$PROJECT_ROOT/filelists/slr_all_train.txt" \
    --validation-files "$PROJECT_ROOT/filelists/slr_all_val.txt" \
    -bs 32 --grad-accumulation 2 \
    --optimizer lamb -lr 0.1 \
    --weight-decay 1e-6 \
    --grad-clip-thresh 1000.0 \
    --epochs 1000 --epochs-per-checkpoint 5 \
    --warmup-steps 1000 \
    --dur-predictor-loss-scale 0.1 \
    --pitch-predictor-loss-scale 0.1 \
    --text-cleaners ipa_cleaners \
    --symbol-set ipa_es \
    --n-speakers 1 --p-arpabet 0.0 \
    --load-pitch-from-disk \
    --validation-freq 5 \
    --num-workers 4 \
    2>&1 | tee "$OUT_DIR/logs/train.log"
```

Los checkpoints se guardan cada 5 épocas para permitir la reanudación del
entrenamiento ante interrupciones del servidor.

## Inferencia

```bash
cd "$FASTPITCH_ROOT"

python3 inference.py \
    -i "$PROJECT_ROOT/examples/frases_ipa.txt" \
    -o "$PROJECT_ROOT/outputs/inference_baseline" \
    --fastpitch "$OUT_DIR/FastPitch_checkpoint_600.pt" \
    --hifigan "$HIFIGAN_PATH" \
    --hifigan-config "$HIFIGAN_CONFIG" \
    --symbol-set ipa_es \
    --text-cleaners ipa_cleaners \
    --p-arpabet 0.0 \
    --affinity disabled \
    --fade-out 0 \
    --cuda \
    --save-mels
```

El archivo `examples/frases_ipa.txt` debe contener una frase IPA por línea,
generada previamente con `espeak-ng` mediante `phonemizer`. Deben preservarse
las marcas de acento.

FastPitch puede reordenar internamente las frases por longitud durante la
inferencia. Por ello, el índice de un audio generado no necesariamente
coincide con el número de línea original sin aplicar el mapeo correspondiente.

## Selección del checkpoint

El checkpoint operativo del Baseline FastPitch se seleccionó mediante
evaluación objetiva sobre el conjunto de validación de 1,216 frases. Esta
evaluación consideró tres métricas complementarias: MCD, F0 RMSE y correlación
de F0. En la ejecución final de la tesis se seleccionó `ckpt_600`, por
presentar el mejor balance entre distorsión espectral y coherencia prosódica.

| Checkpoint | N | MCD ↓ | F0 RMSE ↓ | F0 Corr ↑ |
|---:|---:|---:|---:|---:|
| ckpt_100 | 1216 | 28.764 ± 0.168 | 74.931 ± 2.871 | 0.4862 ± 0.0108 |
| ckpt_200 | 1216 | 27.112 ± 0.159 | 70.684 ± 2.742 | 0.5157 ± 0.0104 |
| ckpt_300 | 1216 | 26.184 ± 0.151 | 67.923 ± 2.681 | 0.5438 ± 0.0101 |
| ckpt_400 | 1216 | 25.612 ± 0.148 | 65.407 ± 2.603 | 0.5624 ± 0.0098 |
| ckpt_500 | 1216 | 25.284 ± 0.145 | 63.718 ± 2.579 | 0.5746 ± 0.0095 |
| **ckpt_600** | **1216** | **25.123 ± 0.142** | **62.812 ± 2.554** | **0.5811 ± 0.0093** |
| ckpt_700 | 1216 | 25.207 ± 0.143 | 63.246 ± 2.567 | 0.5783 ± 0.0094 |
| ckpt_800 | 1216 | 25.391 ± 0.146 | 64.105 ± 2.589 | 0.5712 ± 0.0096 |
| ckpt_900 | 1216 | 25.684 ± 0.149 | 65.338 ± 2.631 | 0.5639 ± 0.0098 |
| ckpt_1000 | 1216 | 25.917 ± 0.153 | 66.214 ± 2.674 | 0.5574 ± 0.0100 |

La tabla completa corresponde a la evaluación de checkpoints reportada en los
anexos de la tesis.

## Resultados sobre test independiente

Una vez seleccionado el checkpoint operativo `ckpt_600`, el Baseline FastPitch
se evaluó sobre un conjunto test independiente de 20 frases, no utilizado
durante entrenamiento ni validación.

| Métrica | Valor |
|---|---:|
| MCD | 25.65 ± 4.47 dB |
| F0 RMSE | 65.87 ± 5.09 Hz |
| F0 Corr | 0.614 ± 0.049 |

Estos valores corresponden a la evaluación final sobre el conjunto test
independiente. No deben confundirse con la evaluación de checkpoints sobre el
conjunto de validación de 1,216 frases.

## Archivos fuera del repositorio

No se deben versionar checkpoints, audios, tensores, logs, descargas ni
directorios de salida. En particular:

```text
*.pt
*.wav
*.zip
logs/
outputs/
```

El repositorio conserva únicamente el código adaptado y su documentación.
