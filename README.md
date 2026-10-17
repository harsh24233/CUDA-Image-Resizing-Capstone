# GPU-Accelerated Image Resizing with CUDA and NVIDIA NPP

A C++/CUDA command-line application that resizes 8-bit grayscale PGM (`P5`) images on an NVIDIA GPU using NVIDIA Performance Primitives (NPP). The project is designed as an original image-resizing capstone; it is not copied from an image-rotation implementation.

**Execution status:** The source and validation workflow are provided, but this package was prepared in an environment without `nvcc`/an accessible NVIDIA GPU. Therefore it contains no invented successful-run logs, benchmark timings, or GPU identity. Build and run the suite in a CUDA-enabled environment before claiming execution proof.

## Features

- NPP linear-interpolation resize (`nppiResize_8u_C1R`).
- CLI arguments for input, output, and positive scale factor.
- P5 binary PGM input/output with max sample value 255.
- CUDA device reporting and CUDA-event timing around the NPP resize call.
- Makefile, single-run script, multi-scale validation script, output validation, benchmark CSV generation, and proof ZIP packaging.
- Optional 100-megapixel test-image generator for a larger execution trial.

## Repository layout

```text
cuda-image-resizing-capstone/
├── src/image_resize.cu             # CLI, PGM I/O, CUDA/NPP pipeline
├── include/pgm_image.h             # Shared image data structure
├── data/sample.pgm                 # Small included test image
├── scripts/run_validation_suite.sh # Multi-scale real-GPU test and logging
├── scripts/generate_large_pgm.py  # Optional 100 MB PGM generator
├── scripts/package_proof.sh        # Package actual run artifacts
├── tests/validate_pgm.py           # Validate PGM payload/dimensions
├── PROJECT_DESCRIPTION.md          # Written project explanation
├── RUBRIC_CHECKLIST.md              # Submission audit checklist
├── Makefile
├── run.sh
└── INSTALL.md
```

## Requirements

- Linux, Bash, Python 3, and `make`.
- NVIDIA GPU with a working driver and CUDA access.
- NVIDIA CUDA Toolkit including `nvcc` and the NPP development headers/libraries.

GitHub hosts the source but does not itself run CUDA code. Use a CUDA-enabled lab or server for execution proof.

## Build

```bash
make
```

The executable is created at `bin/image_resize`. If CUDA is installed outside `/usr/local/cuda`:

```bash
make clean
make CUDA_HOME=/path/to/cuda
```

If the linker cannot find an NPP library, inspect the libraries shipped with your CUDA version and adjust `NPP_LIBS` in the Makefile based on the actual installation. Keep the real compiler/linker output in `logs/build.log` when testing.

## Run once

```bash
./bin/image_resize --input=data/sample.pgm --output=results/sample_50.pgm --scale=0.5
```

Or:

```bash
bash run.sh data/sample.pgm results/sample_50.pgm 0.5
```

CLI options use the `--name=value` format:

- `--input=FILE.pgm`: input P5 image.
- `--output=FILE.pgm`: output path.
- `--scale=FACTOR`: positive finite scale; `0.5` halves dimensions, `2.0` doubles them.

Dimensions are rounded to the nearest integer. The output width and height must each be at least one pixel. This version does not support PNG/JPEG, RGB images, or ASCII PGM (`P2`).

## Run validation and collect proof

For five scales on the included image:

```bash
bash scripts/run_validation_suite.sh
```

For the same tests plus a generated 100-megapixel grayscale image (about 100 MB of pixel data) resized at 0.5 scale:

```bash
bash scripts/run_validation_suite.sh --large
```

The large test needs enough host RAM and GPU memory. The script writes environment/build logs under `logs/`, output images and measured timings under `results/`, and checks output dimensions/pixel counts. It refuses to add a timing to the CSV if the program did not report one. When the suite finishes successfully, package the actual artifacts:

```bash
bash scripts/package_proof.sh
```

This creates `submission_artifacts/proof_of_execution.zip`. Review its contents before upload. Never add manually invented GPU names, timings, or success logs. The separate PGM validation utility can be run without CUDA:

```bash
python3 tests/validate_pgm.py data/sample.pgm
```

## Timing interpretation

The reported GPU time surrounds the NPP resize call using CUDA events. It excludes file I/O, allocation, and host/device copies, so it must not be described as end-to-end runtime. Report the GPU model, toolkit version, dimensions, scales, and measured values from your actual run logs.

## Troubleshooting

- `nvcc: command not found`: load/install the CUDA Toolkit or correct `CUDA_HOME`.
- Missing `nppi_geometry_transforms.h`: check the CUDA/NPP development installation.
- Undefined NPP symbol: confirm the installed NPP library containing the resize API and update Makefile link flags if necessary.
- No CUDA device: use an environment that exposes an NVIDIA GPU and driver.
- Invalid input: ensure the file is a binary `P5` PGM with max value `255`.

## Submission checklist

See [`RUBRIC_CHECKLIST.md`](RUBRIC_CHECKLIST.md) for a point-by-point audit. The code repository, execution-proof artifact, and project description can be prepared here; the public repository URL and real GPU execution proof still need to be produced by the submitter. The 5–10 minute presentation is a separate rubric component and is not included in this package.
