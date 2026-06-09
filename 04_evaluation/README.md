# 05_evaluation — Evaluación objetiva y subjetiva

## Propósito

Esta carpeta contiene los scripts, plantillas y resultados tabulados usados
para evaluar los modelos TTS de la tesis. La evaluación se divide en:

1. selección objetiva de checkpoints sobre validación;
2. evaluación objetiva final sobre test independiente;
3. evaluación subjetiva MOS/CMOS;
4. análisis complementarios y generación de figuras.

## Modelos evaluados

- **Baseline FastPitch:** referencia interna adaptada al español mediante IPA,
  con predictor de pitch determinista.
- **DDPM+BETO:** modelo propuesto, con predictor de pitch DDPM y
  condicionamiento semántico BETO.

CoquiTTS aparece únicamente como referencia perceptual externa en CMOS. No fue
entrenado como parte de esta tesis.

## Métricas objetivas

| Métrica | Descripción | Mejor valor |
|---|---|---|
| MCD | Distorsión espectral entre referencia y síntesis | Menor |
| F0 RMSE | Error cuadrático medio del contorno de pitch | Menor |
| F0 Corr | Correlación del contorno de pitch | Mayor |

## Selección de checkpoints

La selección se realizó sobre el conjunto de validación completo de **1,216
frases**. Esta etapa escoge el punto operativo; no corresponde al test final.

### Baseline FastPitch

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

El checkpoint `ckpt_600` se seleccionó por presentar el mejor balance entre
distorsión espectral y coherencia prosódica sobre validación.

### DDPM+BETO

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

El checkpoint `ckpt_500` se seleccionó por presentar el mejor balance entre
menor MCD, menor F0 RMSE y mayor correlación de F0.

Los valores tabulados también están disponibles en `results/`.

## Resultados sobre test independiente

Una vez seleccionados los checkpoints operativos, ambos modelos fueron
evaluados sobre un conjunto test independiente de **20 frases**, no utilizado
durante entrenamiento ni validación.

| Modelo | MCD ↓ | F0 RMSE ↓ | F0 Corr ↑ |
|---|---:|---:|---:|
| Baseline FastPitch | 25.65 ± 4.47 | 65.87 ± 5.09 | 0.614 ± 0.049 |
| DDPM+BETO | 22.60 ± 3.45 | 50.72 ± 3.94 | 0.718 ± 0.043 |

Estos valores corresponden a la evaluación objetiva final sobre test
independiente. No deben confundirse con las tablas de selección de checkpoints
sobre validación.

## Evaluación subjetiva

La evaluación perceptual se documenta en `subjective/`:

- **MOS:** escala de 1 a 5 para valorar naturalidad, inteligibilidad y calidad
  general de cada audio.
- **CMOS:** escala de -3 a +3 para comparación pairwise entre DDPM+BETO y
  CoquiTTS con el mismo texto.

En CMOS, los valores positivos favorecen DDPM+BETO, los negativos favorecen
CoquiTTS y cero indica ausencia de preferencia perceptual.

## Resultados de evaluación subjetiva

### MOS — DDPM+BETO

| Dimensión | Media |
|---|---:|
| Naturalidad | 3.267 |
| Inteligibilidad | 4.617 |
| Calidad general | 3.917 |

DDPM+BETO obtuvo una puntuación alta en inteligibilidad y una valoración
favorable de calidad general. La naturalidad recibió una puntuación menor, lo
que señala limitaciones en el ritmo, las pausas y la expresividad global.

### CMOS — DDPM+BETO vs CoquiTTS

| Métrica | Valor |
|---|---:|
| Media CMOS | 1.55 |
| Mediana | 2.0 |
| IC95% | [1.11, 1.99] |
| Wilcoxon | W = 207.0, p = 3.04 × 10⁻⁷ |
| Preferencia por DDPM+BETO | 80% |

La escala CMOS va de -3 a +3. Los valores positivos indican preferencia por
DDPM+BETO, los negativos indican preferencia por CoquiTTS y cero representa
ausencia de preferencia perceptual.

CoquiTTS se utilizó como referencia perceptual externa, no como ablación
arquitectónica. Bajo el protocolo evaluado, DDPM+BETO fue preferido frente a
CoquiTTS; este resultado no implica superioridad general frente a CoquiTTS en
otros corpus, voces o condiciones experimentales.

## Scripts disponibles

| Script | Función |
|---|---|
| `objective/detect_audio_mapping.py` | Detecta el reordenamiento de audios producido por FastPitch durante inferencia. |
| `objective/compute_metrics.py` | Calcula MCD, F0 RMSE y F0 Corr. |
| `objective/checkpoint_selection.py` | Evalúa checkpoints y ayuda a seleccionar el punto operativo. |
| `objective/stratified_analysis.py` | Análisis complementario por tipo de estructura prosódica. |
| `objective/generate_figures.py` | Genera figuras comparativas de espectrogramas y pitch. |
| `subjective/analyze_subjective_results.py` | Analiza respuestas de cuestionarios MOS/CMOS. |

Ejemplo:

```bash
python objective/compute_metrics.py --help
python subjective/analyze_subjective_results.py respuestas.csv --mode mos
```

Los scripts objetivos esperan los Mels de referencia bajo
`data/features/mels/`, en concordancia con `01_preprocessing`.

## Notas importantes

- Las constantes originales de 10 frases se conservan como utilidades
  auxiliares e históricas.
- La evaluación final reportada usa un test independiente de 20 frases.
- Los resultados oficiales son los presentados en este README y en
  `results/`.
- No se versionan audios, Mels, embeddings, checkpoints, outputs ni logs.
