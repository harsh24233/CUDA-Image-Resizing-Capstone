#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 <input.pgm> <output.pgm> <scale>"
  echo "Example: $0 data/sample.pgm results/sample_50.pgm 0.5"
  exit 2
fi

INPUT="$1"
OUTPUT="$2"
SCALE="$3"

if [[ ! -f "$INPUT" ]]; then
  echo "Error: input file not found: $INPUT" >&2
  exit 2
fi

mkdir -p "$(dirname "$OUTPUT")"
make
./bin/image_resize --input="$INPUT" --output="$OUTPUT" --scale="$SCALE"
