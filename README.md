# TTS Español — FastPitch con DDPM y BETO

Sistema de síntesis de voz en español basado en FastPitch, extendido con un
predictor estocástico de pitch mediante DDPM y condicionamiento semántico
mediante BETO. El objetivo es mejorar el modelado prosódico del habla
sintetizada en español, manteniendo una arquitectura modular, reproducible y
controlable.

Este proyecto no desarrolla un sistema TTS desde cero. Adapta FastPitch al
español mediante transcripciones IPA y propone **DDPM+BETO** como extensión
acústico-prosódica. Para la generación de forma de onda se utiliza un vocoder
HiFi-GAN universal preentrenado, mantenido fijo para todos los modelos.

## Contribuciones principales

1. **Adaptación de FastPitch al español mediante IPA**  
   Se incorporó un vocabulario `ipa_es` y un flujo de fonemización para usar
   transcripciones fonéticas del español como entrada del modelo acústico.

2. **Pipeline de preprocesamiento acústico y textual**  
   Se construyó un flujo para normalizar texto, convertir a IPA, procesar
   audios a 22,050 Hz, extraer pitch, preparar filelists y generar
   representaciones semánticas.

3. **Condicionamiento semántico mediante BETO**  
   El texto ortográfico original se procesa con BETO para obtener una
   representación contextual `[CLS]` de 768 dimensiones, proyectada a 384
   dimensiones mediante `Linear + LayerNorm + Tanh`.

4. **Predictor estocástico de pitch basado en DDPM**  
   El predictor determinista de pitch de FastPitch se reemplaza por un módulo
   `PitchDiffusion`, entrenado mediante pérdida de denoising.

5. **Evaluación objetiva y subjetiva**  
   Se comparó el Baseline FastPitch contra DDPM+BETO mediante MCD, F0 RMSE,
   F0 Corr, MOS y CMOS.

## Arquitectura general

```text
Texto original + audio del corpus
        ↓
01_preprocessing
        ↓
FastPitch adaptado al español
        ├── Baseline FastPitch
        └── DDPM+BETO
        ↓
Mel-espectrograma
        ↓
HiFi-GAN universal preentrenado
        ↓
Audio sintetizado
```

La rama fonética convierte el texto a IPA para alimentar FastPitch. La rama
semántica conserva el texto ortográfico original para BETO. BETO no reemplaza
la representación IPA, sino que la complementa con contexto semántico.

El módulo DDPM no genera audio: modela y predice pitch por fonema. Finalmente,
HiFi-GAN convierte el Mel generado por el modelo acústico en audio y se
mantiene fijo para todos los modelos comparados.

## Resultados objetivos

La evaluación objetiva final se realizó sobre un conjunto test independiente
de 20 frases, no utilizado durante entrenamiento ni durante la selección de
checkpoints.

| Modelo | Checkpoint | MCD ↓ | F0 RMSE ↓ | F0 Corr ↑ |
|---|---:|---:|---:|---:|
| Baseline FastPitch | `ckpt_600` | 25.65 ± 4.47 | 65.87 ± 5.09 | 0.614 ± 0.049 |
| DDPM+BETO | `ckpt_500` | 22.60 ± 3.45 | 50.72 ± 3.94 | 0.718 ± 0.043 |

DDPM+BETO redujo la distorsión espectral, redujo el error de F0 y aumentó la
correlación del contorno tonal respecto al Baseline FastPitch. Esto indica una
mejora simultánea en fidelidad espectral y coherencia prosódica bajo el
protocolo evaluado.

La selección de checkpoints se realizó previamente sobre el conjunto de
validación completo de 1,216 frases:

- Baseline FastPitch: `ckpt_600`.
- DDPM+BETO: `ckpt_500`.

## Resultados subjetivos

La evaluación perceptual se realizó mediante MOS y CMOS.

### MOS

| Dimensión | Media |
|---|---:|
| Naturalidad | 3.267 |
| Inteligibilidad | 4.617 |
| Calidad general | 3.917 |

Los resultados MOS muestran que DDPM+BETO genera habla altamente inteligible
y con calidad general favorable. La naturalidad obtuvo una puntuación menor,
lo que evidencia limitaciones pendientes en ritmo, pausas y expresividad
global.

### CMOS

El estudio CMOS comparó DDPM+BETO frente a CoquiTTS usando el mismo contenido
textual. La escala fue de -3 a +3, donde los valores positivos indican
preferencia por DDPM+BETO.

| Métrica | Valor |
|---|---:|
| Media CMOS | 1.55 |
| Mediana | 2.0 |
| IC95% | [1.11, 1.99] |
| Wilcoxon | W = 207.0, p = 3.04 × 10⁻⁷ |
| Preferencia por DDPM+BETO | 80% |

Los evaluadores prefirieron DDPM+BETO frente a CoquiTTS bajo el protocolo
aplicado. CoquiTTS se utilizó únicamente como referencia perceptual externa,
no como ablación arquitectónica ni como modelo entrenado en la tesis. Este
resultado no implica superioridad general en otros corpus, voces o condiciones.

## Corpus

Se utilizó un subconjunto experimental de habla en español latinoamericano
derivado de recursos abiertos OpenSLR/Google.

| Elemento | Valor |
|---|---:|
| Audios procesados | 12,160 |
| Duración total | 18.95 horas |
| Entrenamiento | 10,944 frases |
| Validación | 1,216 frases |
| Partición | 90% entrenamiento / 10% validación |
| Dialectos | Argentina, Colombia, Perú, Venezuela y Chile |
| Frecuencia de muestreo | 22,050 Hz |
| Bandas Mel | 80 |

La variante ecuatoriana no fue incluida porque no formaba parte de la
colección abierta seleccionada.

## Estructura del repositorio

```text
tts-espanol-fastpitch/
├── 01_preprocessing/        Preprocesamiento textual/acústico y features
├── 02_fastpitch_baseline/   Adaptación de FastPitch al español mediante IPA
├── 03_modelo_propuesto/     Modelo DDPM+BETO
├── 04_hifigan_vocoder/      Vocoder HiFi-GAN universal preentrenado
├── 05_evaluation/           Evaluación objetiva y subjetiva
└── README.md
```

### `01_preprocessing/`

Preparación del corpus, normalización textual, conversión a IPA, extracción de
pitch, generación de filelists y precomputación de embeddings BETO.

### `02_fastpitch_baseline/`

Modificaciones a FastPitch para soportar `ipa_es` e `ipa_cleaners`:

- `common/text/symbols.py`
- `common/text/cleaners.py`
- `common/text/text_processing.py`
- `prepare_dataset.py`

### `03_modelo_propuesto/`

Implementación de DDPM+BETO:

- `fastpitch/model.py`
- `fastpitch/pitch_diffusion.py`
- `fastpitch/bert_conditioner.py`
- `fastpitch/data_function.py`
- `fastpitch/loss_function.py`
- `train.py`

### `04_hifigan_vocoder/`

Documenta el uso de HiFi-GAN universal preentrenado como vocoder fijo. No
contiene entrenamiento ni ajuste del vocoder.

### `05_evaluation/`

Contiene scripts y tablas para selección de checkpoints, evaluación objetiva,
evaluación subjetiva MOS/CMOS y generación de figuras.

## Instalación

### FastPitch base

```bash
git clone https://github.com/NVIDIA/DeepLearningExamples.git
cd DeepLearningExamples/PyTorch/SpeechSynthesis/FastPitch
```

### Adaptación del baseline

```bash
cp /ruta/tts-espanol-fastpitch/02_fastpitch_baseline/common/text/symbols.py common/text/
cp /ruta/tts-espanol-fastpitch/02_fastpitch_baseline/common/text/cleaners.py common/text/
cp /ruta/tts-espanol-fastpitch/02_fastpitch_baseline/common/text/text_processing.py common/text/
cp /ruta/tts-espanol-fastpitch/02_fastpitch_baseline/prepare_dataset.py .
```

### Modelo DDPM+BETO

```bash
cp /ruta/tts-espanol-fastpitch/03_modelo_propuesto/fastpitch/model.py fastpitch/
cp /ruta/tts-espanol-fastpitch/03_modelo_propuesto/fastpitch/pitch_diffusion.py fastpitch/
cp /ruta/tts-espanol-fastpitch/03_modelo_propuesto/fastpitch/bert_conditioner.py fastpitch/
cp /ruta/tts-espanol-fastpitch/03_modelo_propuesto/fastpitch/data_function.py fastpitch/
cp /ruta/tts-espanol-fastpitch/03_modelo_propuesto/fastpitch/loss_function.py fastpitch/
cp /ruta/tts-espanol-fastpitch/03_modelo_propuesto/train.py .
```

### Dependencias

```bash
pip install torch torchaudio
pip install phonemizer librosa numpy scipy transformers soundfile pandas matplotlib
sudo apt-get install espeak-ng
```

La implementación NVIDIA/FastPitch puede requerir NVIDIA Apex para los
optimizadores fusionados, incluido LAMB. Su instalación depende de las
versiones locales de CUDA, PyTorch y del entorno NVIDIA utilizado.

## Pipeline completo

```bash
# 1. Preprocesamiento
cd 01_preprocessing/
bash scripts/01_download_openslr.sh data/openslr_raw
python scripts/02_organize_and_convert.py
python scripts/03_build_filelists.py
python scripts/04_extract_features.py
python scripts/05_precompute_bert.py
python scripts/06_validate_split.py

# 2. Entrenamiento Baseline FastPitch
# Ver 02_fastpitch_baseline/README.md

# 3. Entrenamiento DDPM+BETO
# Ver 03_modelo_propuesto/README.md

# 4. Vocodificación con HiFi-GAN universal
# Ver 04_hifigan_vocoder/README.md

# 5. Evaluación objetiva y subjetiva
# Ver 05_evaluation/README.md
```

## Infraestructura utilizada

- **Cluster:** HPC CEDIA.
- **GPU:** NVIDIA A100-SXM4-40GB.
- **Framework:** PyTorch.
- **Entorno:** Python 3.10 / Conda.

## Notas de reproducibilidad

- Los audios, checkpoints, embeddings y tensores `.pt` no se incluyen por
  tamaño y licenciamiento.
- El repositorio contiene scripts y modificaciones de código, no el corpus
  completo.
- Los checkpoints deben generarse localmente o ubicarse externamente.
- HiFi-GAN universal debe descargarse por separado.
- Las rutas mostradas son ejemplos y deben adaptarse al entorno local.

## Citación

```bibtex
@thesis{vacacela2026tts,
  title   = {Sistema TTS en español basado en FastPitch con difusión
             generativa y condicionamiento semántico BETO},
  author  = {Vacacela, John},
  school  = {Universidad de Cuenca},
  year    = {2026}
}
```

## Licencia

FastPitch y las partes derivadas del código base pertenecen a NVIDIA y se
rigen por sus licencias originales. La licencia de las modificaciones propias
debe definirse en el archivo `LICENSE` del repositorio.
