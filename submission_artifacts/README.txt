Execution proof is intentionally not pre-filled with invented logs or GPU results.

On a CUDA-enabled NVIDIA environment, run:
  bash scripts/run_validation_suite.sh
  bash scripts/package_proof.sh

The resulting proof_of_execution.zip should contain real build/environment logs,
per-scale run logs, output PGM files, and a benchmark CSV with measured timings.
