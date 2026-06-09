# Modelo 2: FastPitch + DDPM + BETO

## Propósito

Esta carpeta documenta el modelo propuesto de la tesis: **DDPM+BETO**. Parte
del Baseline FastPitch adaptado al español y conserva:

- el **FastPitch Encoder / Codificador fonético**, alimentado con IPA;
- el Adaptador de Varianza y el Decoder Transformer de FastPitch;
- HiFi-GAN como vocoder externo fijo durante la evaluación.

Sobre esa base, el modelo incorpora:

- el **BETO Codificador Semántico** para condicionar la representación
  fonética con información contextual del texto original;
- un **Predictor de Pitch DDPM** que reemplaza el predictor determinista de
  pitch del Baseline FastPitch.

Esta carpeta no contiene checkpoints, audios, embeddings BETO precomputados,
datos del corpus ni outputs de entrenamiento. Solo conserva el código
modificado respecto a NVIDIA FastPitch.

## Relación con la tesis

La implementación corresponde a la **Fase 2 de la metodología**, en el bloque
“Arquitectura Neuronal y Modelado Prosódico”.

La salida del Codificador fonético se combina con el condicionamiento del
BETO Codificador Semántico para formar la representación condicionada
`H_enc'`. Esta representación alimenta `PitchDiffusion`, el Adaptador de
Varianza y el Decoder Transformer.

## Archivos modificados

| Archivo | Función |
|---|---|
| `fastpitch/model.py` | Integra BETO y PitchDiffusion dentro de FastPitch. |
| `fastpitch/pitch_diffusion.py` | Implementa el predictor de pitch DDPM con 100 pasos de difusión. |
| `fastpitch/bert_conditioner.py` | Implementa el condicionamiento semántico BETO y la proyección 768 → 384. |
| `fastpitch/data_function.py` | Modifica el dataset y el collate para cargar información semántica o embeddings BETO. |
| `fastpitch/loss_function.py` | Incorpora `pitch_diff_loss` a la función de pérdida total. |
| `train.py` | Registra `pitch_diff_loss` y permite entrenamiento con inicialización desde checkpoint. |

Estos archivos deben copiarse sobre una instalación funcional de:

```text
DeepLearningExamples/PyTorch/SpeechSynthesis/FastPitch
```

Antes deben aplicarse las modificaciones de `02_fastpitch_baseline`, que
añaden `ipa_es` e `ipa_cleaners`.

## Arquitectura resumida

```text
Texto original
→ BETO / embeddings semánticos
→ vector semántico 768d
→ Linear(768, 384) + LayerNorm(384) + Tanh
→ c_sem 384d
→ suma aditiva con la salida del FastPitch Encoder
→ H_enc'
→ Predictor de Pitch DDPM
→ Adaptador de Varianza
→ Decoder Transformer
→ espectrograma Mel
→ HiFi-GAN externo
```

BETO procesa el texto original para obtener una representación semántica. En
paralelo, FastPitch recibe la secuencia fonética IPA mediante el Camino
fonético. La fusión ocurre antes de la predicción de pitch.

## Condicionamiento semántico BETO

`BERTSemanticConditioner` utiliza:

```text
BETO [CLS] 768d
→ Linear(768, 384)
→ LayerNorm(384)
→ Tanh
→ c_sem 384d
```

- Modelo: `dccuchile/bert-base-spanish-wwm-cased`.
- Representación contextual: vector `[CLS]`.
- Dimensión de BETO: 768.
- Dimensión del encoder FastPitch: 384.
- La proyección permite sumar `c_sem` a la salida del encoder.
- La suma se aplica por broadcasting temporal sobre la secuencia fonética.

En `model.py`, el resultado es la representación condicionada `H_enc'`.

## Predictor de Pitch DDPM

`fastpitch/pitch_diffusion.py` implementa `PitchDiffusion` con:

- 100 pasos de difusión;
- beta schedule lineal `[1e-4, 0.02]`;
- dimensión interna 256;
- embedding temporal sinusoidal de 128 dimensiones;
- 4 bloques `DiffusionStep`;
- predicción de un valor escalar de pitch por fonema.

El DDPM reemplaza el predictor determinista de pitch del baseline. No recibe
BETO como una concatenación adicional por frame: opera sobre `H_enc'`, que ya
contiene el condicionamiento semántico.

Durante el entrenamiento, `p_losses()` añade ruido al pitch objetivo y aprende
a predecir ese ruido. Durante la inferencia, el proceso parte de ruido
gaussiano y aplica los 100 pasos reverse para generar pitch.

El argumento `bert_emb` de `PitchDiffusion` se conserva por compatibilidad de
interfaz, pero no se concatena por frame.

## Función de pérdida

`fastpitch/loss_function.py` extiende la pérdida original:

```text
loss = mel_loss
     + duration_loss
     + pitch_loss
     + pitch_diff_loss
     + attention_loss
     + energy_loss si aplica
```

- `pitch_diff_loss` es la pérdida de denoising del DDPM.
- `mel_loss` evalúa la reconstrucción acústica.
- La pérdida de duración supervisa el predictor de duración.
- La pérdida de atención supervisa el alineamiento interno.
- Si `energy_conditioning` está desactivado, la energía no participa.

`pitch_diff_loss` se suma a la pérdida total y se incluye en `meta`.
`train.py` lo registra durante el entrenamiento.

## Dataset y embeddings BETO

`fastpitch/data_function.py` prepara:

- texto IPA;
- Mel objetivo;
- Pitch/F0;
- energía;
- alineamiento previo;
- embeddings semánticos BETO.

Los embeddings por palabra se buscan mediante:

```bash
export BERT_EMBEDDINGS_DIR=$PROJECT_ROOT/data/bert_word_embeddings
export PITCH_DIR=$PROJECT_ROOT/data/features/pitch
```

Si la variable no está definida, se usa
`data/bert_word_embeddings`. La ruta absoluta del entorno experimental CEDIA
fue eliminada.

Los filelists DDPM+BETO incluyen el texto original en la cuarta columna:

```text
wav_path|text_orig|ipa_text|text_orig
```

Cuando `--load-pitch-from-disk` está activo, `data_function.py` admite dos
formas de localizar el pitch:

- si la segunda columna es una ruta `.pt` existente, la usa directamente para
  conservar compatibilidad con filelists FastPitch antiguos;
- en el formato de este repositorio, resuelve `{id}.pt` dentro de `PITCH_DIR`,
  cuyo valor por defecto es `data/features/pitch`.

### Nota de integración

Esta instantánea de `data_function.py` alinea embeddings precomputados por
palabra con los tokens IPA y los entrega como `bert_padded`. Sin embargo,
`model.py` desempaqueta ese tensor pero el condicionamiento global solo se
aplica cuando el llamador proporciona `inputs_bert`.

Por tanto, al integrar estos archivos en la base FastPitch se debe verificar
que el frontend de entrenamiento o inferencia entregue a `inputs_bert` la
representación global de 768 dimensiones esperada. Esta observación documenta
el estado del código sin cambiar la arquitectura ni decidir entre el camino
global `[CLS]` y el tensor alineado por tokens.

## Entrenamiento

```bash
export PROJECT_ROOT=/ruta/al/proyecto
export FASTPITCH_ROOT=$PROJECT_ROOT/external/FastPitch
export DATA_DIR=$PROJECT_ROOT/data
export OUT_DIR=$PROJECT_ROOT/outputs/ddpm_beto
export BASE_CKPT=$PROJECT_ROOT/outputs/baseline_fastpitch/FastPitch_checkpoint_600.pt
export BERT_EMBEDDINGS_DIR=$PROJECT_ROOT/data/bert_word_embeddings
export PITCH_DIR=$PROJECT_ROOT/data/features/pitch

cd "$FASTPITCH_ROOT"
mkdir -p "$OUT_DIR/logs"

torchrun --nproc_per_node=2 train.py \
    --cuda --amp \
    -o "$OUT_DIR" \
    --log-file "$OUT_DIR/logs/train.log" \
    --dataset-path "$DATA_DIR" \
    --training-files "$PROJECT_ROOT/filelists/slr_bert_train.txt" \
    --validation-files "$PROJECT_ROOT/filelists/slr_bert_val.txt" \
    -bs 32 --grad-accumulation 2 \
    --optimizer lamb -lr 0.1 \
    --weight-decay 1e-6 \
    --grad-clip-thresh 1000.0 \
    --epochs 800 --epochs-per-checkpoint 5 \
    --warmup-steps 1000 \
    --dur-predictor-loss-scale 0.1 \
    --pitch-predictor-loss-scale 0.1 \
    --text-cleaners ipa_cleaners \
    --symbol-set ipa_es \
    --n-speakers 1 --p-arpabet 0.0 \
    --load-pitch-from-disk \
    --init-from-checkpoint "$BASE_CKPT" \
    --validation-freq 5 \
    --num-workers 4 \
    2>&1 | tee "$OUT_DIR/logs/train.log"
```

El modelo se inicializa desde el checkpoint operativo del Baseline FastPitch.
En la tesis, DDPM+BETO se entrenó hasta 800 épocas. Los checkpoints se guardan
cada 5 épocas para facilitar la recuperación ante interrupciones.

## Selección del checkpoint

El checkpoint operativo del modelo DDPM+BETO se seleccionó mediante evaluación
objetiva sobre el conjunto de validación de 1,216 frases. Esta evaluación
consideró MCD, F0 RMSE y correlación de F0. En la ejecución final de la tesis
se seleccionó `ckpt_500`, por presentar el mejor balance entre distorsión
espectral y coherencia prosódica.

| Checkpoint | N | MCD ↓ | F0 RMSE ↓ | F0 Corr ↑ |
|---:|---:|---:|---:|---:|
| ckpt_100 | 1216 | 27.436 ± 1.612 | 66.982 ± 2.731 | 0.5573 ± 0.0106 |
| ckpt_200 | 1216 | 25.981 ± 1.553 | 61.247 ± 2.642 | 0.6138 ± 0.0098 |
| ckpt_300 | 1216 | 24.927 ± 1.506 | 57.364 ± 2.551 | 0.6542 ± 0.0091 |
| ckpt_400 | 1216 | 24.216 ± 1.472 | 54.906 ± 2.486 | 0.6815 ± 0.0084 |
| **ckpt_500** | **1216** | **23.981 ± 1.447** | **53.821 ± 2.428** | **0.6919 ± 0.0080** |
| ckpt_600 | 1216 | 24.073 ± 1.451 | 54.186 ± 2.439 | 0.6884 ± 0.0081 |
| ckpt_700 | 1216 | 24.295 ± 1.463 | 55.028 ± 2.462 | 0.6797 ± 0.0085 |
| ckpt_800 | 1216 | 24.584 ± 1.489 | 56.317 ± 2.506 | 0.6689 ± 0.0088 |

La tabla corresponde a la evaluación de checkpoints reportada en los anexos
de la tesis.

## Resultados sobre test independiente

Una vez seleccionado el checkpoint operativo `ckpt_500`, el modelo DDPM+BETO
se evaluó sobre un conjunto test independiente de 20 frases, no utilizado
durante entrenamiento ni validación.

| Métrica | Valor |
|---|---:|
| MCD | 22.60 ± 3.45 dB |
| F0 RMSE | 50.72 ± 3.94 Hz |
| F0 Corr | 0.718 ± 0.043 |

Estos valores corresponden a la evaluación final sobre el conjunto test
independiente. No deben confundirse con la evaluación de checkpoints sobre el
conjunto de validación de 1,216 frases.

## Inferencia

```bash
cd "$FASTPITCH_ROOT"

python3 inference.py \
    -i "$PROJECT_ROOT/examples/frases_ipa.txt" \
    -o "$PROJECT_ROOT/outputs/inference_ddpm_beto" \
    --fastpitch "$OUT_DIR/FastPitch_checkpoint_500.pt" \
    --hifigan "$PROJECT_ROOT/external/hifigan/g_02500000" \
    --hifigan-config "$PROJECT_ROOT/external/hifigan/config.json" \
    --symbol-set ipa_es \
    --text-cleaners ipa_cleaners \
    --p-arpabet 0.0 \
    --affinity disabled \
    --fade-out 0 \
    --cuda \
    --save-mels
```

La entrada debe contener la representación IPA requerida por FastPitch. El
condicionamiento semántico también necesita el texto original o los embeddings
BETO correspondientes, según la versión adaptada de `inference.py`. El script
de inferencia original de NVIDIA debe extenderse para entregar `inputs_bert`;
ese archivo no forma parte de esta carpeta.

## Reproducibilidad

Para reproducir el entrenamiento:

1. Aplicar primero las modificaciones de `02_fastpitch_baseline`.
2. Copiar los archivos de esta carpeta sobre la misma base FastPitch.
3. Preparar los filelists DDPM+BETO.
4. Precomputar pitch y embeddings BETO; definir `PITCH_DIR` y
   `BERT_EMBEDDINGS_DIR`.
5. Verificar que entrenamiento e inferencia entreguen `inputs_bert`.
6. Inicializar desde el checkpoint del Baseline FastPitch.

## Archivos fuera del repositorio

No se deben versionar:

- checkpoints `.pt` o `.pth`;
- audios `.wav`;
- embeddings BETO `.pt`, `.npy` o `.npz`;
- logs;
- outputs de entrenamiento;
- datasets;
- archivos comprimidos.
