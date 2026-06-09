# HiFi-GAN universal vocoder

## Propósito

Esta carpeta documenta la etapa de vocodificación utilizada en la tesis. En
los experimentos finales no se entrenó un vocoder propio; se utilizó un
checkpoint universal preentrenado de HiFi-GAN como componente fijo para
convertir los mel-espectrogramas generados por Baseline FastPitch y DDPM+BETO
en señales de audio.

Mantener el vocoder fijo permitió atribuir las diferencias observadas entre
modelos al modelado acústico-prosódico, no a variaciones en la generación de
forma de onda.

## Rol dentro del sistema

```text
Mel-espectrograma generado por Baseline FastPitch o DDPM+BETO
        ↓
HiFi-GAN universal preentrenado
        ↓
Audio sintetizado (.wav)
```

## Qué se hizo

- Se utilizó HiFi-GAN universal preentrenado.
- El vocoder se mantuvo fijo durante la evaluación.
- Se aplicó el mismo vocoder a Baseline FastPitch y DDPM+BETO.
- Se trabajó con mel-espectrogramas de 80 bandas.
- La frecuencia de muestreo fue 22,050 Hz.
- El rango Mel utilizado fue 0–8,000 Hz.
- El hop length fue 256.
- La ventana FFT fue 1024.

## Qué no se hizo

- No se entrenó HiFi-GAN desde cero.
- No se hizo fine-tuning de HiFi-GAN.
- No se compararon distintos vocoders.
- No se usó el vocoder como variable experimental.
- No se atribuyeron mejoras al vocoder.

## Archivos

| Archivo | Función |
|---|---|
| `inference_e2e.py` | Script auxiliar para convertir mel-espectrogramas en audio usando un checkpoint HiFi-GAN. |
| `config_v1.json` | Configuración compatible con HiFi-GAN v1 y los parámetros acústicos usados en la tesis. |

El script se integra con una instalación del repositorio oficial HiFi-GAN y
utiliza sus módulos `env.py` y `models.py`.

## Uso básico

```bash
export HIFIGAN_DIR=/ruta/a/hifigan
export HIFIGAN_CHECKPOINT=$HIFIGAN_DIR/g_02500000
export INPUT_MELS=/ruta/a/mels_generados
export OUTPUT_WAVS=/ruta/a/audios_generados

cd "$HIFIGAN_DIR"

python /ruta/al/proyecto/04_hifigan_vocoder/inference_e2e.py \
    --input_mels_dir "$INPUT_MELS" \
    --output_dir "$OUTPUT_WAVS" \
    --checkpoint_file "$HIFIGAN_CHECKPOINT"
```

`inference_e2e.py` espera archivos Mel `.npy`. También espera que el archivo
`config.json` compatible con el checkpoint se encuentre en el mismo directorio
que `HIFIGAN_CHECKPOINT`. `config_v1.json` documenta la configuración
acústica usada en la tesis y puede copiarse como `config.json` en la
instalación local cuando corresponda al checkpoint utilizado.

El checkpoint HiFi-GAN no se incluye en el repositorio por tamaño y
licenciamiento. Debe descargarse o ubicarse externamente según la instalación
local.

## Relación con la tesis

En la tesis, HiFi-GAN corresponde a la **Fase 3: Generación de Audio y
Vocodificación**. Su función es transformar el espectrograma Mel generado por
el modelo acústico en una señal de audio sintetizada.

Como el mismo vocoder se utilizó para todos los modelos comparados, la
evaluación se centra en las diferencias del modelo acústico-prosódico.
