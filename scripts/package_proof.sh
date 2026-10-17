#!/usr/bin/env bash
# Package only artifacts that actually exist; does not fabricate execution proof.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ! -s logs/build.log || ! -s logs/environment.log || ! -s results/benchmark.csv ]]; then
  echo "Missing build/environment logs or benchmark CSV. Run scripts/run_validation_suite.sh first." >&2
  exit 2
fi
if ! grep -q 'Status: SUCCESS' logs/scale_050.log 2>/dev/null; then
  echo "Expected successful run log logs/scale_050.log is missing; refusing to package incomplete proof." >&2
  exit 2
fi
mkdir -p submission_artifacts
files=(logs/build.log logs/environment.log logs/scale_*.log results/sample_scale_*.pgm results/benchmark.csv)
for candidate in logs/large_*.log logs/large_output_validation.log results/large_test_scale_*.pgm; do
  [[ -e "$candidate" ]] && files+=("$candidate")
done
# Note: Exclude raw 100MB input data/large_test.pgm from the archive to ensure the ZIP
# remains well within LMS/submission upload limits while retaining output images and logs.
if command -v zip >/dev/null 2>&1; then
  zip -q -r submission_artifacts/proof_of_execution.zip "${files[@]}"
else
  python3 -c "import sys, zipfile; from pathlib import Path; out = Path('submission_artifacts/proof_of_execution.zip'); [zf.write(f, f) for zf in [zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED)] for f in sys.argv[1:] if Path(f).exists()]" "${files[@]}"
fi
sha256sum submission_artifacts/proof_of_execution.zip > submission_artifacts/SHA256SUMS.txt
printf 'Created %s\n' "$PWD/submission_artifacts/proof_of_execution.zip"
