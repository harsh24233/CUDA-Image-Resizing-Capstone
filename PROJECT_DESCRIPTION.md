# Project Description: GPU-Accelerated Image Resizing with CUDA and NVIDIA NPP

## 1. Problem Statement and Project Objectives

Digital image resizing (spatial scaling) is a fundamental operation across computer vision pipelines, deep learning preprocessing, video transcoding, and digital asset rendering. Resizing high-resolution images on a CPU using sequential or multi-threaded CPU routines frequently creates throughput bottlenecks due to memory bandwidth limits and cache contention when dealing with millions of pixels.

The primary objective of this project is to implement, validate, and benchmark a high-performance command-line application that offloads 8-bit grayscale image resizing to an NVIDIA GPU. Rather than reimplementing basic resampling kernels from scratch, the project leverages **NVIDIA Performance Primitives (NPP)**—NVIDIA's highly optimized library of image and signal processing functions—integrated with **NVIDIA CUDA** for device management, memory orchestration, and precise execution profiling.

---

## 2. Why GPU Acceleration is Useful for Image Resizing

Image resizing is an **embarrassingly parallel** workload. Every destination pixel's color intensity is mathematically independent of all other destination pixels and can be computed concurrently:

- **Massive Thread-Level Parallelism (SIMT):** Modern NVIDIA GPUs feature thousands of CUDA cores grouped into Streaming Multiprocessors (SMs). An image with millions of pixels (e.g., 100 megapixels) can map individual destination pixels or small tiles to independent GPU threads, saturating the GPU's execution units.
- **High Memory Bandwidth:** Discrete GPUs provide hundreds of gigabytes per second (or terabytes per second) of VRAM bandwidth (GDDR6/HBM), allowing fast parallel fetching of source pixel neighborhoods.
- **Latency Hiding:** While CPU threads incur high penalties on cache misses when striding across large 2D image arrays, GPUs hide memory access latency by context-switching between warps at hardware speed.
- **Hardware-Level Interpolation:** GPU architectures contain dedicated texture and spatial interpolation hardware, which NPP exploits to achieve near-peak memory-bandwidth efficiency during image resampling.

---

## 3. Role of CUDA and NPP

- **NVIDIA CUDA Runtime (`cuda_runtime.h`):** Provides the low-level foundation for querying hardware device properties (GPU architecture, compute capability, VRAM size), allocating linear device memory buffers (`cudaMalloc`), orchestrating bi-directional host-device data transfers (`cudaMemcpy`), synchronizing execution, and profiling kernel latency via hardware timers (`cudaEvent_t`).
- **NVIDIA Performance Primitives (`npp.h`, `nppi.h`, `nppi_geometry_transforms.h`):** Supplies production-grade, highly tuned computer vision routines. Specifically, the function `nppiResize_8u_C1R` executes 1-channel 8-bit unsigned integer 2D geometric resizing using hardware-optimized bilinear interpolation, handling boundary conditions, spatial alignment, and coalesced memory access patterns.

---

## 4. Image Resizing Algorithm: Bilinear Interpolation

The application applies **Bilinear Interpolation** (`NPPI_INTER_LINEAR`), which samples the four nearest source pixels around the fractional mapped coordinate and computes an area-weighted average.

### Mathematical Formulation

1. **Continuous Coordinate Mapping:**  
   Given a source image of size $W_{src} \times H_{src}$ and a scaled destination of size $W_{dst} \times H_{dst}$, each discrete destination pixel center $(x_d, y_d)$ where $0 \le x_d < W_{dst}$ and $0 \le y_d < H_{dst}$ is mapped back to continuous source coordinates $(x_s, y_s)$:
   $$x_s = (x_d + 0.5) \cdot \frac{W_{src}}{W_{dst}} - 0.5$$
   $$y_s = (y_d + 0.5) \cdot \frac{H_{src}}{H_{dst}} - 0.5$$

2. **Identifying the Four Neighboring Pixels:**  
   The four surrounding integer grid coordinates are:
   $$x_1 = \lfloor x_s \rfloor, \quad x_2 = x_1 + 1$$
   $$y_1 = \lfloor y_s \rfloor, \quad y_2 = y_1 + 1$$
   *(Clamped to $[0, W_{src}-1]$ and $[0, H_{src}-1]$ at boundaries).*

3. **Weight Computation:**  
   The fractional offsets representing horizontal and vertical distances are:
   $$\alpha = x_s - x_1, \quad \beta = y_s - y_1 \quad (\text{where } \alpha, \beta \in [0, 1])$$

4. **2D Weighted Linear Combination:**  
   The interpolated pixel intensity $I_{dst}(x_d, y_d)$ is calculated by:
   $$I_{dst}(x_d, y_d) = (1 - \alpha)(1 - \beta) \cdot I_{src}(x_1, y_1) + \alpha (1 - \beta) \cdot I_{src}(x_2, y_1) + (1 - \alpha)\beta \cdot I_{src}(x_1, y_2) + \alpha \beta \cdot I_{src}(x_2, y_2)$$
   The resulting value is rounded and clamped to the 8-bit unsigned integer interval $[0, 255]$.

---

## 5. End-to-End Processing Pipeline

The execution flow consists of 8 strictly ordered steps:

```text
[Input PGM File]
       │
       ▼
1. CLI Argument Parsing & Validation (--input, --output, --scale)
       │
       ▼
2. Binary PGM (P5) File Reading & Header Parsing
       │
       ▼
3. Output Dimension Calculation: round(W * scale), round(H * scale)
       │
       ▼
4. CUDA Device Query & Hardware Capability Detection
       │
       ▼
5. Device Memory Allocation (cudaMalloc) & Host-to-Device Copy (cudaMemcpyHostToDevice)
       │
       ▼
6. NPP Bilinear Resize (nppiResize_8u_C1R) profiled via CUDA Events
       │
       ▼
7. Device-to-Host Copy (cudaMemcpyDeviceToHost) & Device Memory Deallocation (cudaFree)
       │
       ▼
8. Binary PGM (P5) File Writing & Validation
       │
       ▼
[Output Resized PGM File]
```

### Timing Scope
CUDA events (`cudaEventRecord` and `cudaEventElapsedTime`) profile **strictly the GPU resize execution** (`nppiResize_8u_C1R`) and exclude disk I/O, memory allocations, and host-device transfer times. This ensures accurate profiling of GPU kernel compute performance rather than general system latency.

---

## 6. Important Implementation Decisions

1. **Binary PGM (P5) Format:**  
   The binary Netpbm grayscale format (`P5`) is chosen because it stores raw 8-bit unsigned bytes directly without compression artifacts or complex third-party container parsing, allowing deterministic byte-level validation.
2. **Rounding Policy:**  
   Output dimensions are computed using `std::llround(dimension * scale)`, which implements half-away-from-zero rounding. Scaled dimensions $< 1$ are strictly rejected to prevent invalid zero-size buffers.
3. **Structured Argument Interface:**  
   Command-line parameters require explicit `--name=value` syntax (`--input=`, `--output=`, `--scale=`). Unrecognized options or non-finite scales (e.g., negative, NaN, infinity) immediately terminate execution with exit code 2.
4. **Honest Proof & Reproducibility:**  
   The repository separates verification scripts from actual execution logs. It explicitly refuses to hardcode or fabricate benchmark figures or GPU device strings.

---

## 7. Memory Management and Error Handling

- **Dual-Layer Resource Tracking:** Host pixel buffers utilize RAII-managed `std::vector<std::uint8_t>`, ensuring automatic cleanup upon stack unwinding. Device buffers utilize `cudaMalloc` and are explicitly released with `cudaFree` on all exit paths.
- **CUDA Event Management:** Events (`start_event`, `stop_event`) are tracked with an allocation flag (`events_created`) and destroyed via `cudaEventDestroy` before process termination.
- **Strict Error Wrappers:**
  - `CheckCuda(cudaError_t, const char*)`: Inspects every runtime API call and formats human-readable messages via `cudaGetErrorString`.
  - `CheckNpp(NppStatus, const char*)`: Validates the return code of NPP operations, reporting error status codes upon failure.
- **Fail-Fast Philosophy:** If no CUDA-capable device is detected (`cudaGetDeviceCount(&count) <= 0`), the application immediately reports an informative error and aborts with exit code 1, rather than silently falling back to a slow CPU routine.

---

## 8. How to Compile and Run

### Compilation
Using GNU Make:
```bash
make clean
make
```
*(If CUDA is installed in a non-default path: `make CUDA_HOME=/path/to/cuda`)*.

### Single Execution
```bash
./bin/image_resize --input=data/sample.pgm --output=results/sample_50.pgm --scale=0.5
```
Or via the shell wrapper:
```bash
bash run.sh data/sample.pgm results/sample_50.pgm 0.5
```

### Comprehensive Validation Suite
```bash
bash scripts/run_validation_suite.sh
# For the optional 100-megapixel stress test:
bash scripts/run_validation_suite.sh --large
```

### Packaging Proof
```bash
bash scripts/package_proof.sh
```

---

## 9. Testing Methodology and Validation Criteria

Validation scripts verify both mathematical correctness and system robustness:

1. **Dimensional Accuracy:** The output header dimensions $(W_{out}, H_{out})$ must match $\text{round}(W_{in} \cdot \text{scale})$ and $\text{round}(H_{in} \cdot \text{scale})$.
2. **Payload Completeness:** The exact binary byte count following the single whitespace separator must strictly equal $W_{out} \times H_{out}$ bytes.
3. **Multi-Scale Testing:** The test suite evaluates scaling factors across both downsampling and upsampling: `0.5`, `0.75`, `1.0`, `1.5`, and `2.0`.
4. **Stress Testing:** `scripts/generate_large_pgm.py` produces a synthetic $10,000 \times 10,000$ (100 megapixel, 100 MB payload) PGM image to verify buffer handling, memory limits, and compute throughput on large payloads.
5. **CLI Robustness:** Rejection of non-existent files, invalid scale factors (`0`, `-1.5`, `abc`), and unparseable arguments.

---

## 10. Experimental Results & Verification Status

The validation suite was executed directly on an NVIDIA GPU environment with verified authentic outputs:

### Hardware & Environment Profile
- **GPU Model:** NVIDIA GeForce RTX 4050 Laptop GPU
- **Compute Capability:** 8.9 (Ada Lovelace architecture)
- **Host System:** Ubuntu 24.04 LTS (WSL2 on Windows 11)
- **NVIDIA Driver:** 592.82 (CUDA 13.1 driver capability)
- **CUDA Toolkit Compiler:** `nvcc` release 12.0, V12.0.140

### Measured Benchmark Results
The application was evaluated across multiple downsampling and upsampling factors using `data/sample.pgm` ($320 \times 240$ input, 76,800 bytes). Output files and byte-level pixel dimensions were validated mathematically:

| Input Image | Scale | Input Res | Output Res | Output Bytes | GPU Resize Time (ms) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `data/sample.pgm` | **0.50** | $320 \times 240$ | $160 \times 120$ | 19,200 | **9.07 ms** | SUCCESS |
| `data/sample.pgm` | **0.75** | $320 \times 240$ | $240 \times 180$ | 43,200 | **14.49 ms** | SUCCESS |
| `data/sample.pgm` | **1.00** | $320 \times 240$ | $320 \times 240$ | 76,800 | **5.82 ms** | SUCCESS |
| `data/sample.pgm` | **1.50** | $320 \times 240$ | $480 \times 360$ | 172,800 | **5.70 ms** | SUCCESS |
| `data/sample.pgm` | **2.00** | $320 \times 240$ | $640 \times 480$ | 307,200 | **6.29 ms** | SUCCESS |

All output images exist in `results/`, execution logs are saved in `logs/`, and authentic evidence is packaged in `submission_artifacts/proof_of_execution.zip`.

---

## 11. Challenges, Lessons, and Future Extensions

### Key Challenges & Lessons
- **API Signature Precision:** NPP function signatures require exact ordering of parameters—specifically distinguishing the line step pitch (`int nSrcStep`) from image dimensions (`NppiSize oSrcSize`).
- **Timing Interpretation:** Profiling device kernels via `cudaEvent` accurately measures GPU execution time, but real-world system latency is dominated by PCIe transfer times and disk I/O on smaller images.
- **Pitch Alignment:** While contiguous linear memory (`cudaMalloc`) functions for arbitrary widths, production NPP deployments benefit from 2D pitched allocations (`cudaMallocPitch` or `nppiMalloc_8u_C1R`) to ensure 32-byte or 128-byte hardware memory alignment.

### Future Improvements
1. **Multi-Channel & Color Support:** Extend support to 3-channel (RGB) images (`nppiResize_8u_C3R`) and PNG/JPEG formats via image codecs.
2. **CPU vs. GPU Benchmark Comparison:** Implement a multi-threaded CPU reference (e.g., OpenMP or OpenCV CPU resize) to calculate empirical speedup ratios across varying resolutions.
3. **Pitched Memory Allocation:** Integrate `nppiMalloc_8u_C1R` and `nppiFree` to utilize hardware-aligned line steps.
4. **Batch Processing & CUDA Streams:** Process multiple images concurrently using asynchronous memory transfers across independent CUDA streams.
