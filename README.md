# GPU Image Processing Capstone — CUDA NPP Image Resizing

## Overview

This project is a high-performance GPU-accelerated image processing application developed as the **GPU Specialization Capstone Project**.

The application demonstrates how NVIDIA CUDA and the **NVIDIA Performance Primitives (NPP)** library can be utilized to perform 2D image resizing on the GPU. The program accepts an 8-bit grayscale image in binary Netpbm PGM (`P5`) format, transfers pixel data from host memory to device memory, executes spatial resampling using NPP's geometric transformation primitives with hardware-optimized bilinear interpolation, transfers the processed image back to the host, and saves the verified output as a valid PGM image.

The application supports arbitrary positive scaling factors through a command-line interface, validates input/output dimensions, ensures memory safety, and measures GPU execution time using hardware-level CUDA events.

The project demonstrates practical mastery of:
- **NVIDIA CUDA C++** runtime programming
- **NVIDIA Performance Primitives (NPP)** image geometry operations
- GPU memory management (`cudaMalloc`, `cudaMemcpy`, `cudaFree`)
- Parallel bilinear interpolation and coordinate mapping
- Hardware-accurate profiling with CUDA events (`cudaEvent_t`)
- Robust command-line argument parsing and input validation
- Automated build systems (`Makefile`) and multi-scale execution scripts (`run.sh`)

---

## Video Demonstration

Watch the project demonstration video here:

*(Paste your recorded video link here, e.g. Google Drive or YouTube Unlisted)*

---

## Project Motivation & Why GPU Acceleration

Image resizing is a critical preprocessing operation across computer vision pipelines, convolutional neural networks, video transcoding, and graphics rendering. 

In digital image resampling, every destination pixel can be computed independently of all other destination pixels. While a conventional CPU implementation processes pixels through serialized nested loops or multi-threaded CPU threads constrained by memory bus contention and cache thrashing, GPU acceleration exploits **Single Instruction, Multiple Threads (SIMT)** execution:
- **Massive Thread Parallelism:** Modern NVIDIA GPUs contain thousands of CUDA cores grouped into Streaming Multiprocessors (SMs), enabling simultaneous calculation of millions of destination pixels.
- **High Memory Bandwidth:** Discrete GPUs provide hundreds of gigabytes per second of high-speed VRAM bandwidth (GDDR6), allowing fast coalesced fetching of source pixel neighborhoods.
- **Latency Hiding:** The GPU hardware scheduler context-switches warps instantaneously to mask memory fetch latency.
- **Hardware-Level Spatial Interpolation:** NVIDIA NPP harnesses spatial cache locality and hardware texture interpolation units to deliver maximum compute efficiency.

---

## Project Goals

1. Process image data directly on NVIDIA GPU hardware.
2. Utilize production-grade CUDA and NPP libraries rather than standard CPU multi-threading.
3. Implement high-quality image resizing using NVIDIA NPP (`nppiResize_8u_C1R`).
4. Manage bi-directional memory transfers between host and device with complete error checking.
5. Apply mathematical bilinear interpolation across continuous coordinate space.
6. Support configurable scaling factors via a structured command-line interface.
7. Measure exact GPU kernel execution time using CUDA events.
8. Validate output dimensions, header syntax, and exact byte-level payloads.
9. Provide automated build automation (`Makefile`) and execution scripts (`run.sh`).
10. Demonstrate multi-scale validation across downsampling and upsampling factors (`0.5`, `0.75`, `1.0`, `1.5`, `2.0`).

---

## Technologies Used

| Technology | Purpose |
| :--- | :--- |
| **C++14** | Core application logic and Netpbm PGM I/O |
| **NVIDIA CUDA** | GPU runtime, device memory management, and CUDA event profiling |
| **CUDA NPP** | High-performance image geometry transformations |
| **`nppiResize_8u_C1R`** | 1-channel 8-bit unsigned integer bilinear image resizing |
| **`nvcc`** | NVIDIA CUDA Compiler driver |
| **GNU Make** | Automated compilation and build management |
| **Bash & Python 3** | Automated multi-scale testing, validation, and benchmarking |
| **PGM (P5)** | Binary 8-bit grayscale image format |

---

## Hardware & Execution Environment

The application was executed, profiled, and benchmarked on an authentic NVIDIA GPU environment:

```text
GPU: NVIDIA GeForce RTX 4050 Laptop GPU
Compute Capability: 8.9 (Ada Lovelace architecture)
Global Memory: 6,141 MiB GDDR6
Driver Version: 592.82
CUDA Driver Version: 13.1
CUDA Compiler: nvcc release 12.0, V12.0.140
Operating System: Ubuntu 24.04 LTS (WSL2 on Windows 11)
```

---

## Application Architecture

The end-to-end processing pipeline consists of the following 8 stages:

```text
               Input PGM Image (P5)
                        │
                        ▼
             ┌─────────────────────┐
             │ CLI Option Parsing  │ (--input, --output, --scale)
             └─────────────────────┘
                        │
                        ▼
             ┌─────────────────────┐
             │  Host Image Loading │ (Verify magic 'P5', dims, maxval 255)
             └─────────────────────┘
                        │
                        ▼
             ┌─────────────────────┐
             │ Calculate Output    │ scaled_w = round(W * scale)
             │ Dimensions          │ scaled_h = round(H * scale)
             └─────────────────────┘
                        │
                        ▼
             ┌─────────────────────┐
             │ Device Allocation   │ cudaMalloc() & cudaMemcpyHostToDevice
             │ & Upload to GPU     │
             └─────────────────────┘
                        │
                        ▼
             ┌─────────────────────┐
             │ NVIDIA NPP Resize   │ nppiResize_8u_C1R()
             │ (Bilinear Resample) │ Profiled via cudaEventRecord()
             └─────────────────────┘
                        │
                        ▼
             ┌─────────────────────┐
             │ Download from GPU   │ cudaMemcpyDeviceToHost() & cudaFree()
             └─────────────────────┘
                        │
                        ▼
             ┌─────────────────────┐
             │ Save Output PGM     │ Write P5 binary header + raw pixel bytes
             └─────────────────────┘
                        │
                        ▼
              Resized Output Image
```

---

## Image Processing Algorithm: Bilinear Interpolation

The application applies **Bilinear Interpolation** (`NPPI_INTER_LINEAR`), which maps each destination coordinate back into continuous source coordinate space and computes a distance-weighted average of the four nearest neighboring source pixels:

### 1. Continuous Coordinate Mapping
For each discrete destination pixel center $(x_d, y_d)$ in the output image of size $W_{dst} \times H_{dst}$, continuous coordinates $(x_s, y_s)$ in the source image of size $W_{src} \times H_{src}$ are computed via:
$$x_s = (x_d + 0.5) \cdot \frac{W_{src}}{W_{dst}} - 0.5$$
$$y_s = (y_d + 0.5) \cdot \frac{H_{src}}{H_{dst}} - 0.5$$

### 2. Identifying Neighboring Pixels
The four surrounding integer grid points are:
$$x_1 = \lfloor x_s \rfloor, \quad x_2 = x_1 + 1$$
$$y_1 = \lfloor y_s \rfloor, \quad y_2 = y_1 + 1$$
*(Coordinates are clamped to $[0, W_{src}-1]$ and $[0, H_{src}-1]$ to respect image boundaries).*

### 3. Calculating Fractional Weights
$$\alpha = x_s - x_1, \quad \beta = y_s - y_1 \quad (\text{where } \alpha, \beta \in [0, 1])$$

### 4. 2D Linear Combination
The interpolated intensity $I_{dst}(x_d, y_d)$ is calculated as:
$$I_{dst}(x_d, y_d) = (1 - \alpha)(1 - \beta) I_{src}(x_1, y_1) + \alpha (1 - \beta) I_{src}(x_2, y_1) + (1 - \alpha)\beta I_{src}(x_1, y_2) + \alpha \beta I_{src}(x_2, y_2)$$

The resulting intensity is rounded and stored as an 8-bit unsigned byte ($0 \le I_{dst} \le 255$).

---

## CUDA/NPP Processing

The core GPU resizing operation is implemented using NVIDIA NPP:

```cpp
NppStatus nppiResize_8u_C1R(
    const Npp8u * pSrc,
    int           nSrcStep,        // Source line step (stride) in bytes
    NppiSize      oSrcSize,        // Full source image dimensions
    NppiRect      oSrcRectROI,     // Source Region of Interest
    Npp8u       * pDst,
    int           nDstStep,        // Destination line step (stride) in bytes
    NppiSize      oDstSize,        // Full destination image dimensions
    NppiRect      oDstRectROI,     // Destination Region of Interest
    int           eInterpolation   // NPPI_INTER_LINEAR
);
```

### Accurate Hardware Profiling
Kernel execution latency is isolated using hardware CUDA events:
```cpp
cudaEventRecord(start_event, 0);
nppiResize_8u_C1R(device_source, source.width, source_size, source_roi,
                  device_destination, destination.width, destination_size,
                  destination_roi, NPPI_INTER_LINEAR);
cudaEventRecord(stop_event, 0);
cudaEventSynchronize(stop_event);
cudaEventElapsedTime(&elapsed_ms, start_event, stop_event);
```
This isolates the GPU compute interval from disk I/O, memory allocations, and PCIe bus transfers.

---

## Measured Performance & Benchmark Results

The application was tested across five scaling factors on an **NVIDIA GeForce RTX 4050 Laptop GPU** using `data/sample.pgm` ($320 \times 240$ input, 76,800 bytes). All outputs were verified for dimension and byte-count accuracy:

| Input Image | Scale | Input Resolution | Output Resolution | Output Bytes | GPU Resize Time | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `data/sample.pgm` | **0.50** | $320 \times 240$ | $160 \times 120$ | 19,200 bytes | **9.07 ms** | SUCCESS |
| `data/sample.pgm` | **0.75** | $320 \times 240$ | $240 \times 180$ | 43,200 bytes | **14.49 ms** | SUCCESS |
| `data/sample.pgm` | **1.00** | $320 \times 240$ | $320 \times 240$ | 76,800 bytes | **5.82 ms** | SUCCESS |
| `data/sample.pgm` | **1.50** | $320 \times 240$ | $480 \times 360$ | 172,800 bytes | **5.70 ms** | SUCCESS |
| `data/sample.pgm` | **2.00** | $320 \times 240$ | $640 \times 480$ | 307,200 bytes | **6.29 ms** | SUCCESS |

The raw benchmark results are logged in `results/benchmark.csv`.

---

## Repository Structure

```text
CUDA-Image-Resizing-Capstone/
├── proof_of_execution.zip          # Packaged authentic execution proof (root)
├── LICENSE                         # Open-source MIT License
├── Makefile                        # Build automation with nvcc detection
├── run.sh                          # Single-run execution shell script
├── README.md                       # Comprehensive master documentation
├── .gitignore                      # Clean repository rules
├── src/
│   └── image_resize.cu             # Main CUDA/NPP resizing application
├── include/
│   └── pgm_image.h                 # Shared PGM image struct & declarations
├── data/
│   └── sample.pgm                  # Sample 320x240 grayscale input image
├── results/
│   ├── benchmark.csv               # Measured GPU execution benchmark table
│   └── sample_scale_*.pgm          # Resized output images for all scales
├── logs/
│   ├── environment.log             # GPU specs, driver & nvcc versions
│   ├── build.log                   # Compilation log
│   └── scale_*.log                 # Execution logs for each scale factor
├── submission_artifacts/
│   └── proof_of_execution.zip      # Packaged authentic execution proof
├── scripts/
│   ├── run_validation_suite.sh     # Multi-scale execution & logging suite
│   ├── generate_large_pgm.py       # 100-Megapixel synthetic stress generator
│   └── package_proof.sh            # Packages verified artifacts into ZIP
└── tests/
    └── validate_pgm.py             # Standalone PGM header & payload validator
```

---

## Build and Execution Instructions

### Prerequisites
- Linux / WSL2 with Ubuntu
- NVIDIA GPU with compatible driver (CUDA 11+ or 12+)
- NVIDIA CUDA Toolkit (`nvcc`, NPP development libraries)
- GNU Make, GCC/G++, Python 3

### 1. Build
```bash
make clean
make
```
The executable is generated at `bin/image_resize`.

### 2. Single Run via Executable
```bash
./bin/image_resize --input=data/sample.pgm --output=results/sample_50.pgm --scale=0.5
```

### 3. Single Run via Shell Script
```bash
bash run.sh data/sample.pgm results/sample_50.pgm 0.5
```

### 4. Run the Full Multi-Scale Validation Suite
```bash
bash scripts/run_validation_suite.sh
```
This automatically runs all 5 scales, validates the output files, records execution times to `results/benchmark.csv`, and writes logs to `logs/`.

### 5. Optional 100-Megapixel Stress Test
```bash
bash scripts/run_validation_suite.sh --large
```

### 6. Package Proof of Execution
```bash
bash scripts/package_proof.sh
```
This packages all genuine logs, output images, and `benchmark.csv` into `submission_artifacts/proof_of_execution.zip`.

---

## Challenges Encountered & Lessons Learned

1. **NPP Parameter Signature Exactness:**  
   The NPP API requires strict ordering between the line step pitch (`int nSrcStep`) and image dimensions (`NppiSize oSrcSize`). Distinguishing line step from image size resolved earlier compilation hurdles.
2. **Timing Isolation:**  
   Profiling device work using `cudaEventRecord` ensures that measured latency isolates actual GPU kernel execution without skewing from PCIe memory transfers or disk I/O.
3. **Memory Safety & Error Handling:**  
   Host and device buffers are validated with custom wrappers (`CheckCuda`, `CheckNpp`). Device memory is systematically released on all execution branches, guaranteeing leak-free execution without silent CPU fallbacks.

---

## Future Improvements

1. **Multi-Channel & Color Images:** Expand support from single-channel grayscale (`8u_C1R`) to 3-channel RGB images (`8u_C3R`) and modern formats (JPEG, PNG).
2. **CPU vs. GPU Benchmarking:** Implement an OpenMP or OpenCV CPU baseline to calculate empirical speedup ratios.
3. **Asynchronous Stream Concurrency:** Implement CUDA streams to overlap PCIe host-to-device transfers with kernel execution across image batches.
