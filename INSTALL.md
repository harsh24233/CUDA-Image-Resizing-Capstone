# Installation and Build

## Environment

Use a Linux machine or lab session with an NVIDIA GPU, a working driver, CUDA Toolkit, NPP headers/libraries, GNU Make, Bash, and Python 3. Verify access before building:

```bash
nvidia-smi
nvcc --version
```

## Build

```bash
make clean
make
```

The Makefile defaults to `CUDA_HOME=/usr/local/cuda`. Override it if necessary:

```bash
make CUDA_HOME=/your/cuda/path
```

The default link flags are `-lnppig -lnppc -lcudart`. NPP packaging can vary between toolkit versions; use the library files installed with your toolkit to resolve a linker error rather than copying binaries from another machine.

## Execute

```bash
bash run.sh data/sample.pgm results/sample_half.pgm 0.5
```

The command-line executable accepts `--input=`, `--output=`, and `--scale=`. Input must be an 8-bit grayscale binary PGM (`P5`) with maximum value 255.

## Collect evidence

```bash
bash scripts/run_validation_suite.sh --large
bash scripts/package_proof.sh
```

This suite records the actual compiler and GPU environment, build output, run logs, output files, and measured NPP timings. The large option generates a 10000 x 10000 image, which has approximately 100 MB of pixel data; skip it if the available GPU does not have sufficient memory. Inspect the proof ZIP and do not claim results that are not present in the logs.
