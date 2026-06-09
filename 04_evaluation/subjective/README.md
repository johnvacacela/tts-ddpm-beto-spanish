# Evaluación subjetiva

Esta carpeta contiene las plantillas y scripts relacionados con la evaluación
perceptual de los modelos TTS.

## MOS

MOS usa una escala de 1 a 5 para la evaluación individual de audios. Permite
medir naturalidad, inteligibilidad y calidad general.

Resultados oficiales de DDPM+BETO:

| Dimensión | Media |
|---|---:|
| Naturalidad | 3.267 |
| Inteligibilidad | 4.617 |
| Calidad general | 3.917 |

## CMOS

CMOS usa una escala de -3 a +3 para una evaluación comparativa pairwise entre
DDPM+BETO y CoquiTTS usando el mismo texto.

- valores positivos: preferencia por DDPM+BETO;
- valores negativos: preferencia por CoquiTTS;
- cero: ausencia de preferencia perceptual.

| Métrica | Valor |
|---|---:|
| Media CMOS | 1.55 |
| Mediana | 2.0 |
| IC95% | [1.11, 1.99] |
| Wilcoxon | W = 207.0, p = 3.04 × 10⁻⁷ |
| Preferencia por DDPM+BETO | 80% |

CoquiTTS es una referencia perceptual externa, no una ablación arquitectónica
ni un modelo entrenado dentro de la tesis. Los resultados indican preferencia
por DDPM+BETO bajo el protocolo evaluado, no superioridad general.

## Cuestionarios

- `mos_questionnaire.md`
- `cmos_questionnaire.md`

`analyze_subjective_results.py` calcula estadísticos descriptivos e intervalos
de confianza a partir de respuestas exportadas a CSV. Solo requiere la
biblioteca estándar de Python.
