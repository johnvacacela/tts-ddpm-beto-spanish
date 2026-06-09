# Fase 1: Preprocesamiento y Extracción de Características

Esta carpeta reproduce la Fase 1 de la metodología: Preprocesamiento y Extracción de Características.

Su propósito es preparar un corpus abierto de habla en español latinoamericano
para el **Baseline FastPitch** y el modelo **DDPM+BETO**, manteniendo un
**Split 90/10 reproducible** y dos representaciones complementarias:

- **Camino fonético (IPA):** IPA generada con `espeak-ng` mediante
  `phonemizer`, preservando las marcas de acento.
- **Camino semántico (WordPiece):** vector `[CLS]` del
  **BETO Codificador Semántico**
  `dccuchile/bert-base-spanish-wwm-cased`.

## Flujo

```text
Corpus original
→ organización y remuestreo
→ normalización textual
→ conversión a IPA
→ split train/val
→ extracción de características acústicas
→ precomputación BETO
→ validación del split
```

## Estructura

```text
01_preprocessing/
├── README.md
├── scripts/
│   ├── 01_download_openslr.sh
│   ├── 02_organize_and_convert.py
│   ├── 03_build_filelists.py
│   ├── 04_extract_features.py
│   ├── 05_precompute_bert.py
│   └── 06_validate_split.py
├── visualization/
│   └── transformacion_senal.py
└── examples/
    └── .gitkeep
```

`visualization/transformacion_senal.py` genera una figura de apoyo
metodológico. No es un paso obligatorio del pipeline.

## Corpus y configuración

- Fuente: subconjuntos abiertos de OpenSLR/Google.
- Dialectos: Argentina, Colombia, Perú, Venezuela y Chile.
- Subconjunto final reportado: 12,160 audios, aproximadamente 18.95 horas.
- Train: 10,944 frases.
- Validación: 1,216 frases.
- Frecuencia de muestreo: 22,050 Hz.
- Split: 90% entrenamiento y 10% validación, con seed `1234`.

Los conteos anteriores corresponden al corpus final de la tesis. Los scripts
también funcionan con subconjuntos parciales y reportan los conteos obtenidos.

## Requisitos

Sistema:

```bash
sudo apt-get install espeak-ng wget unzip
```

Python:

```bash
pip install librosa soundfile phonemizer numpy scipy torch transformers \
  matplotlib seaborn
```

La precomputación BETO puede ejecutarse en CPU, pero se recomienda una GPU
compatible con PyTorch.

## Ejecución

Ejecute los comandos desde `01_preprocessing/`:

```bash
# 1. Descargar los cinco subconjuntos.
bash scripts/01_download_openslr.sh data/openslr_raw

# 2. Centralizar WAV, remuestrear a 22,050 Hz y generar metadata con IPA.
python scripts/02_organize_and_convert.py \
  --raw_dir data/openslr_raw \
  --wav_dir data/wavs_22k \
  --metadata data/metadata.txt

# 3. Crear un único split y sus dos formatos de filelist.
python scripts/03_build_filelists.py \
  --metadata data/metadata.txt \
  --out_dir filelists

# 4. Precomputar Mels auxiliares, Pitch/F0 y energía RMS.
python scripts/04_extract_features.py \
  --train_file filelists/slr_all_train.txt \
  --val_file filelists/slr_all_val.txt \
  --out_dir data/features

# 5. Precomputar el vector [CLS] de BETO.
python scripts/05_precompute_bert.py \
  --train_file filelists/slr_bert_train.txt \
  --val_file filelists/slr_bert_val.txt \
  --out_dir data/bert_embeddings

# 6. Validar el split.
python scripts/06_validate_split.py \
  --train_file filelists/slr_all_train.txt \
  --val_file filelists/slr_all_val.txt
```

Todos los scripts Python exponen sus opciones con `--help`. Las rutas por
defecto son relativas al directorio desde el cual se ejecuta el comando.

## Salidas

```text
data/
├── openslr_raw/                 corpus descargado
├── wavs_22k/                    audios remuestreados
├── metadata.txt
├── features/
│   ├── mels/{id}.pt             Tensor (80, T)
│   ├── pitch/{id}.pt            Tensor (T,)
│   └── energy/{id}.pt           Tensor (T,)
└── bert_embeddings/{id}.pt      Tensor (768,)

filelists/
├── slr_all_train.txt
├── slr_all_val.txt
├── slr_bert_train.txt
└── slr_bert_val.txt
```

Los audios procesados reciben prefijos `ar_`, `co_`, `pe_`, `ve_` y `cl_`.
Esto evita colisiones entre identificadores iguales de dialectos distintos.

## Formatos

`data/metadata.txt`:

```text
wav_path|text_orig|ipa_text
```

Filelists del **Baseline FastPitch**:

```text
wav_path|text_orig|ipa_text
```

Filelists de **DDPM+BETO**:

```text
wav_path|text_orig|ipa_text|text_orig
```

La cuarta columna repite el texto original de forma intencional. Se mantiene
para compatibilidad con el cargador del Camino semántico (WordPiece). Ambos
pares de filelists se escriben a partir del mismo objeto de split, por lo que
contienen exactamente las mismas muestras en train y validación.

## Notas metodológicas

### Mel-espectrogramas

Los Mel-espectrogramas pueden calcularse dinámicamente durante el entrenamiento
según la implementación de FastPitch. El script `04_extract_features.py`
también los precomputa como características auxiliares para inspección del
corpus, validación del preprocesamiento y evaluación objetiva espectral, por
ejemplo MCD. No se asume que estos tensores sean la única entrada Mel posible
durante el entrenamiento.

### Pitch/F0

El **Pitch/F0 precomputado** usa `librosa.pyin` con rango de `65` a `500 Hz`
por defecto, `frame_length=2048` y el mismo `hop_length` de las demás
características. Los valores indefinidos de pYIN se guardan como `0.0`; esos
ceros representan regiones no sonoras o no vocalizadas.

### Energía RMS

La energía RMS se guarda como característica auxiliar para análisis, recorte,
control y validación. No se documenta como predictor prosódico explícito de la
arquitectura final.

### BETO

`05_precompute_bert.py` tokeniza el texto con WordPiece y guarda el vector
`[CLS]` de 768 dimensiones producido por
`dccuchile/bert-base-spanish-wwm-cased`. Esto evita ejecutar BETO en cada paso
de entrenamiento.

### Validación y reproducibilidad

El split se genera una sola vez con seed `1234`, estratificado por el prefijo
de dialecto. `06_validate_split.py` reporta:

- número total de muestras y porcentajes train/val;
- balance aproximado por dialecto;
- distribución de longitud de frases y prueba KS;
- cobertura fonética de validación respecto a entrenamiento.

La prueba KS verifica si train y validación presentan distribuciones de
longitud similares. Un valor `p > 0.05` indica que no se detecta una diferencia
estadísticamente significativa.

