#!/usr/bin/env bash
# Run real GPU tests at several scales; optional --large adds a 100 MB input.
set -euo pipefail
cd "$(dirname "$0")/.."
LARGE_TEST=0
if [[ "${1:-}" == "--large" ]]; then LARGE_TEST=1; elif [[ $# -gt 0 ]]; then echo "Usage: $0 [--large]" >&2; exit 2; fi
mkdir -p results logs submission_artifacts

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found; use a CUDA-enabled NVIDIA environment." >&2; exit 2
fi
if ! command -v nvcc >/dev/null 2>&1; then
  echo "ERROR: nvcc not found; install/load the CUDA Toolkit first." >&2; exit 2
fi
make clean
make 2>&1 | tee logs/build.log
{
  echo "=== Timestamp (UTC) ==="; date -u '+%Y-%m-%dT%H:%M:%SZ'
  echo "=== GPU ==="; nvidia-smi
  echo "=== CUDA compiler ==="; nvcc --version
} 2>&1 | tee logs/environment.log
CSV=results/benchmark.csv
printf 'input,scale,input_width,input_height,output_width,output_height,gpu_time_ms,status\n' > "$CSV"
for scale in 0.5 0.75 1.0 1.5 2.0; do
  tag=$(awk -v s="$scale" 'BEGIN {printf "%03d", s*100}'); output="results/sample_scale_${tag}.pgm"; log="logs/scale_${tag}.log"
  echo "=== Running scale=${scale} ===" | tee "$log"
  ./bin/image_resize --input=data/sample.pgm --output="$output" --scale="$scale" 2>&1 | tee -a "$log"
  python3 - "$output" "$scale" "$CSV" "$log" <<'PY'
import re, sys
from pathlib import Path
out, scale, csv, log = sys.argv[1:]
data = Path(out).read_bytes()
# Parse the P5 header, allowing comments before the pixel payload.
def tokens(blob):
    i, values = 0, []
    while len(values) < 4:
        while i < len(blob) and blob[i] in b' \t\r\n': i += 1
        if i < len(blob) and blob[i] == 35:
            end = blob.find(b'\n', i); i = len(blob) if end < 0 else end + 1; continue
        start = i
        while i < len(blob) and blob[i] not in b' \t\r\n#': i += 1
        if start == i: raise SystemExit('Invalid PGM header')
        values.append(blob[start:i])
    if i >= len(blob) or blob[i] not in b' \t\r\n': raise SystemExit('Missing PGM separator')
    i += 1
    return values, blob[i:]
header, pixels = tokens(data)
if header[0] != b'P5' or header[3] != b'255': raise SystemExit('Expected 8-bit P5 PGM')
w, h = int(header[1]), int(header[2])
if len(pixels) != w*h: raise SystemExit(f'Pixel count mismatch: {len(pixels)} != {w*h}')
text = Path(log).read_text(errors='replace')
m = re.search(r'GPU resize time \(ms\):\s*([0-9.eE+-]+)', text)
if not m: raise SystemExit(f'No GPU timing in {log}; refusing to invent benchmark data')
inw, inh = 320, 240
if (w, h) != (round(inw*float(scale)), round(inh*float(scale))): raise SystemExit('Unexpected output dimensions')
with open(csv, 'a', encoding='utf-8') as f: f.write(f'data/sample.pgm,{scale},{inw},{inh},{w},{h},{m.group(1)},SUCCESS\n')
print(f'Validated {out}: {w}x{h}, {len(pixels)} pixel bytes')
PY
done
if [[ "$LARGE_TEST" -eq 1 ]]; then
  python3 scripts/generate_large_pgm.py --output data/large_test.pgm 2>&1 | tee logs/large_data_generation.log
  output=results/large_test_scale_050.pgm; log=logs/large_test_scale_050.log
  echo '=== Running 100-megapixel input at scale=0.5 ===' | tee "$log"
  ./bin/image_resize --input=data/large_test.pgm --output="$output" --scale=0.5 2>&1 | tee -a "$log"
  python3 tests/validate_pgm.py "$output" | tee logs/large_output_validation.log
  python3 - "$log" "$CSV" <<'PY'
import re, sys
from pathlib import Path
log_path, csv_path = sys.argv[1:]
text = Path(log_path).read_text(errors='replace')
m = re.search(r'GPU resize time \(ms\):\s*([0-9.eE+-]+)', text)
if not m: raise SystemExit('No GPU timing in large-data log; refusing to invent benchmark data')
with open(csv_path, 'a', encoding='utf-8') as f:
    f.write(f'data/large_test.pgm,0.5,10000,10000,5000,5000,{m.group(1)},SUCCESS\n')
PY
fi
echo 'Validation suite completed. Review logs/ and results/ before submission.'
