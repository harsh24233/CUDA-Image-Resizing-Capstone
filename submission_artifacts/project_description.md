# GPU-Accelerated Image Resizing using NVIDIA CUDA NPP
**Author:** Harsh  
**Course:** GPU Specialization Capstone Project  
**Target Hardware:** NVIDIA GeForce RTX 4050 Laptop GPU (Compute Capability 8.9, 6141 MiB GDDR6 VRAM)

---

## 1. Project Purpose and Objectives
Digital image resizing (spatial resampling) is a fundamental preprocessing operation across computer vision pipelines, deep learning model training, video transcoding, and graphics rendering. In high-throughput industrial pipelines, resizing large high-resolution images on a CPU creates severe bottlenecks due to serialized pixel traversal, memory bandwidth limits, and cache contention.

The purpose of this capstone project is to demonstrate end-to-end GPU acceleration for 2D image resizing using NVIDIA CUDA and the NVIDIA Performance Primitives (NPP) library. By moving the image manipulation pipeline to dedicated GPU hardware, the application exploits Massive Thread Parallelism (SIMT architecture) and high GDDR6 memory bandwidth to resample destination pixels independently and concurrently.

---

## 2. Technical Implementation Architecture
The project is implemented in modern C++ and compiled with the NVIDIA CUDA Compiler (`nvcc`). It uses a modular, maintainable architecture conforming to the Google C++ Style Guide:

- **Image I/O Subsystem (`include/pgm_image.h`):** A lightweight, self-contained Netpbm PGM (`P5` format, 8-bit single-channel grayscale) parser and writer. It parses image headers, handles comments robustly, validates byte sizes, and avoids external heavyweight third-party dependencies.
- **CLI Subsystem (`src/image_resize.cu`):** Accepts structured command-line parameters (`--input=<path>`, `--output=<path>`, `--scale=<factor>`), performs bounds and format validation, and outputs structured status logs.
- **GPU Core Execution (`src/image_resize.cu`):** Interfaces with the CUDA Runtime and NVIDIA NPP image geometry library (`nppig`, `nppc`), utilizing `nppiResize_8u_C1R` for high-performance bilinear interpolation.

---

## 3. GPU Processing Pipeline and Memory Management
The execution flow strictly adheres to the standard GPU offloading lifecycle:
1. **Host-Side Ingestion:** Loads the source PGM image from disk into host memory and verifies dimensions ($W_{in} \times H_{in}$) and 8-bit grayscale format.
2. **Device Detection:** Queries `cudaGetDeviceProperties` to report hardware specs (GPU name, Compute Capability, total VRAM, and streaming multiprocessors).
3. **GPU Memory Allocation:** Allocates contiguous 2D device buffers using `cudaMalloc` for both input and output images.
4. **Host-to-Device (H2D) Transfer:** Transfers raw pixel data from host RAM to GPU VRAM via `cudaMemcpy(..., cudaMemcpyHostToDevice)`.
5. **NPP Kernel Execution:** Calculates output dimensions ($W_{out} = \text{round}(W_{in} \times s)$, $H_{out} = \text{round}(H_{in} \times s)$), configures source and destination regions of interest (`NppiRect`), and invokes `nppiResize_8u_C1R` using `NPPI_INTER_LINEAR` (bilinear interpolation).
6. **Hardware Timing:** Records exact GPU execution latency using hardware CUDA events (`cudaEventRecord`, `cudaEventElapsedTime`).
7. **Device-to-Host (D2H) Transfer:** Copies the processed output buffer back to host memory via `cudaMemcpy(..., cudaMemcpyDeviceToHost)`.
8. **Output Writing & Cleanup:** Writes the resulting image to disk as a standard PGM file and deallocates device memory via `cudaFree` and event handles via `cudaEventDestroy`.

---

## 4. Resizing Algorithm & Bilinear Interpolation
The spatial transform maps continuous output coordinates back into input coordinate space:
$$x_{in} = (x_{out} + 0.5) \times \frac{W_{in}}{W_{out}} - 0.5, \quad y_{in} = (y_{out} + 0.5) \times \frac{H_{in}}{H_{out}} - 0.5$$
For each mapped continuous point $(x_{in}, y_{in})$, the four nearest discrete neighbor pixels $P(x_1, y_1), P(x_2, y_1), P(x_1, y_2), P(x_2, y_2)$ are sampled. The interpolated intensity value $I(x_{out}, y_{out})$ is calculated using weighted fractional distances $(\alpha = x_{in} - x_1, \beta = y_{in} - y_1)$:
$$I = (1 - \alpha)(1 - \beta)P(x_1, y_1) + \alpha(1 - \beta)P(x_2, y_1) + (1 - \alpha)\beta P(x_1, y_2) + \alpha\beta P(x_2, y_2)$$
NVIDIA NPP executes this resampling across thousands of CUDA threads simultaneously, optimizing cache locality and utilizing texture/memory caching hardware.

---

## 5. Challenges Encountered and Resolutions
1. **NPP API Parameter Constraints:** The `nppiResize_8u_C1R` API expects exact row step parameters (stride in bytes) and `NppiRect` ROI descriptors. An incorrect ROI offset or step parameter caused invalid boundary calculations. This was resolved by computing exact pitch strides (`source.width` in bytes) matching continuous memory blocks.
2. **Cross-Platform Build Tooling:** Developing across Windows and Linux required reliable build automation. While PowerShell lacks native `make`, WSL2 Ubuntu 24.04 provided direct CUDA hardware access via NVIDIA WSL drivers. The `Makefile` was updated with dynamic `which nvcc` detection and fallback linker flags (`-lnppig -lnppc -lcudart`) to compile cleanly across both native Linux and WSL environments.
3. **Artifact Packaging Limits:** Generating 100-megapixel stress test data resulted in large files that exceeded Coursera submission limits. The packaging script (`scripts/package_proof.sh`) was refined to package only the essential logs, benchmark metrics, and verified output PGM images into a compact 70 KB archive.

---

## 6. Experimental Results and Performance Analysis
The project was tested and benchmarked on physical hardware (**NVIDIA GeForce RTX 4050 Laptop GPU**, Driver 592.82, CUDA 12.0) across five representative scaling factors on a $320 \times 240$ input image:

| Test Case | Scale Factor | Input Dimensions | Output Dimensions | Output Size (Bytes) | GPU Execution Time (ms) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Downsampling 1 | 0.50x | 320 x 240 | 160 x 120 | 19,200 | 9.07 ms | SUCCESS |
| Downsampling 2 | 0.75x | 320 x 240 | 240 x 180 | 43,200 | 14.49 ms | SUCCESS |
| Identity | 1.00x | 320 x 240 | 320 x 240 | 76,800 | 5.82 ms | SUCCESS |
| Upsampling 1 | 1.50x | 320 x 240 | 480 x 360 | 172,800 | 5.70 ms | SUCCESS |
| Upsampling 2 | 2.00x | 320 x 240 | 640 x 480 | 307,200 | 6.29 ms | SUCCESS |

All generated output images were verified using byte-level validation scripts (`tests/validate_pgm.py`), ensuring intact headers and accurate file payloads.

---

## 7. Lessons Learned & Key Takeaways
1. **Optimized Libraries vs. Hand-Written Kernels:** Leveraging NVIDIA NPP primitives allows developers to achieve near-optimal hardware performance with robust boundary checking without having to hand-tune warp shuffle instructions or shared memory tile caching.
2. **Accurate GPU Timing:** CPU-side wall clock timers (`std::chrono`) introduce PCIe synchronization and OS scheduling noise. Using asynchronous CUDA events (`cudaEvent_t`) is required for measuring true device kernel execution.
3. **Memory Safety:** Explicit validation of image bounds before GPU allocation prevents out-of-memory kernel launches and segmentation faults.

---

## 8. Limitations and Future Improvements
- **Multi-Channel (RGB) Support:** Currently supports 8-bit single-channel grayscale (`_8u_C1R`). Future work will add 3-channel RGB (`nppiResize_8u_C3R`) and 4-channel RGBA (`nppiResize_8u_C4R`).
- **Additional Interpolation Modes:** Support for cubic spline (`NPPI_INTER_CUBIC`) and Lanczos filtering.
- **Asynchronous Batch Processing:** Integrating CUDA Streams (`cudaStream_t`) to overlap H2D/D2H transfers with kernel computation across image batches.
