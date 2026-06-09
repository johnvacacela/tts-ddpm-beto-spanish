#!/usr/bin/env bash
set -euo pipefail

# Descarga los subconjuntos femeninos de OpenSLR 61 usados en el proyecto.
# Uso: bash scripts/01_download_openslr.sh [directorio_salida]

OUTPUT_DIR="${1:-data/openslr_raw}"
BASE_URL="https://www.openslr.org/resources/61"
DIALECTS=(es_ar es_co es_pe es_ve es_cl)

mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"

for dialect in "${DIALECTS[@]}"; do
    archive="$OUTPUT_DIR/${dialect}_female.zip"
    destination="$OUTPUT_DIR/$dialect"

    echo "Descargando $dialect..."
    wget -c "$BASE_URL/${dialect}_female.zip" -O "$archive"

    echo "Descomprimiendo $dialect..."
    mkdir -p "$destination"
    unzip -q -o "$archive" -d "$destination"
    rm -f "$archive"
done

echo
echo "Descarga completa en: $OUTPUT_DIR"
for dialect in "${DIALECTS[@]}"; do
    count="$(find "$OUTPUT_DIR/$dialect" -type f -name "*.wav" 2>/dev/null | wc -l)"
    echo "  $dialect: $count audios"
done
