#include <cuda_runtime.h>
#include <npp.h>
#include <nppi.h>
#include <nppi_geometry_transforms.h>

#include <cmath>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "pgm_image.h"

namespace {

struct Options {
  std::string input_path;
  std::string output_path;
  double scale = 0.5;
};

bool ParseOptions(int argc, char** argv, Options* options) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    const std::size_t equals = arg.find('=');
    if (equals == std::string::npos) {
      std::cerr << "Invalid argument (expected --name=value): " << arg << '\n';
      return false;
    }
    const std::string key = arg.substr(0, equals);
    const std::string value = arg.substr(equals + 1);
    if (key == "--input") {
      options->input_path = value;
    } else if (key == "--output") {
      options->output_path = value;
    } else if (key == "--scale") {
      try {
        std::size_t consumed = 0;
        options->scale = std::stod(value, &consumed);
        if (consumed != value.size()) {
          std::cerr << "Invalid scale value: " << value << '\n';
          return false;
        }
      } catch (const std::exception&) {
        std::cerr << "Invalid scale value: " << value << '\n';
        return false;
      }
    } else {
      std::cerr << "Unknown argument: " << key << '\n';
      return false;
    }
  }

  if (options->input_path.empty() || options->output_path.empty() ||
      !std::isfinite(options->scale) || options->scale <= 0.0) {
    std::cerr << "Usage: image_resize --input=FILE.pgm "
                 "--output=FILE.pgm --scale=FACTOR\n"
              << "Scale must be a finite positive number.\n";
    return false;
  }
  return true;
}

bool CheckCuda(cudaError_t status, const char* operation) {
  if (status == cudaSuccess) return true;
  std::cerr << operation << " failed: " << cudaGetErrorString(status) << '\n';
  return false;
}

bool CheckNpp(NppStatus status, const char* operation) {
  if (status == NPP_SUCCESS) return true;
  std::cerr << operation << " failed with NPP status "
            << static_cast<int>(status) << '\n';
  return false;
}

bool ReadToken(std::istream& input, std::string* token) {
  token->clear();
  while (input) {
    input >> std::ws;
    if (input.peek() == '#') {
      std::string comment;
      std::getline(input, comment);
      continue;
    }
    break;
  }
  return static_cast<bool>(input >> *token);
}

}  // namespace

bool ReadPgm(const std::string& path, PgmImage* image, std::string* error) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    *error = "Could not open input file: " + path;
    return false;
  }

  std::string magic;
  std::string width_token;
  std::string height_token;
  std::string max_token;
  if (!ReadToken(input, &magic) || magic != "P5" ||
      !ReadToken(input, &width_token) ||
      !ReadToken(input, &height_token) ||
      !ReadToken(input, &max_token)) {
    *error = "Expected a valid binary PGM (P5) header.";
    return false;
  }

  try {
    image->width = std::stoi(width_token);
    image->height = std::stoi(height_token);
    const int max_value = std::stoi(max_token);
    if (image->width <= 0 || image->height <= 0 || max_value != 255) {
      *error = "PGM must have positive dimensions and max value 255.";
      return false;
    }
  } catch (const std::exception&) {
    *error = "Invalid numeric value in PGM header.";
    return false;
  }

  // Consume the single whitespace character separating the header and pixels.
  const int separator = input.get();
  if (separator == EOF || !std::isspace(static_cast<unsigned char>(separator))) {
    *error = "Missing PGM header/pixel separator.";
    return false;
  }

  const std::uint64_t pixel_count =
      static_cast<std::uint64_t>(image->width) * image->height;
  if (pixel_count > static_cast<std::uint64_t>(
                        std::numeric_limits<std::size_t>::max())) {
    *error = "Image dimensions are too large.";
    return false;
  }

  image->pixels.resize(static_cast<std::size_t>(pixel_count));
  input.read(reinterpret_cast<char*>(image->pixels.data()),
             static_cast<std::streamsize>(image->pixels.size()));
  if (input.gcount() != static_cast<std::streamsize>(image->pixels.size())) {
    *error = "Input image pixel data is incomplete.";
    return false;
  }
  return true;
}

bool WritePgm(const std::string& path, const PgmImage& image,
              std::string* error) {
  std::ofstream output(path, std::ios::binary);
  if (!output) {
    *error = "Could not open output file: " + path;
    return false;
  }
  output << "P5\n" << image.width << ' ' << image.height << "\n255\n";
  output.write(reinterpret_cast<const char*>(image.pixels.data()),
               static_cast<std::streamsize>(image.pixels.size()));
  if (!output) {
    *error = "Failed while writing output image: " + path;
    return false;
  }
  return true;
}

int main(int argc, char** argv) {
  Options options;
  if (!ParseOptions(argc, argv, &options)) return 2;

  PgmImage source;
  std::string error;
  if (!ReadPgm(options.input_path, &source, &error)) {
    std::cerr << "Input error: " << error << '\n';
    return 1;
  }

  const long long scaled_width =
      std::llround(static_cast<double>(source.width) * options.scale);
  const long long scaled_height =
      std::llround(static_cast<double>(source.height) * options.scale);
  if (scaled_width < 1 || scaled_height < 1 ||
      scaled_width > std::numeric_limits<int>::max() ||
      scaled_height > std::numeric_limits<int>::max()) {
    std::cerr << "Scale factor produces invalid output dimensions.\n";
    return 2;
  }

  PgmImage destination;
  destination.width = static_cast<int>(scaled_width);
  destination.height = static_cast<int>(scaled_height);
  const std::uint64_t output_pixels =
      static_cast<std::uint64_t>(destination.width) * destination.height;
  if (output_pixels > static_cast<std::uint64_t>(
                          std::numeric_limits<std::size_t>::max())) {
    std::cerr << "Output image is too large.\n";
    return 2;
  }
  destination.pixels.resize(static_cast<std::size_t>(output_pixels));

  int device_count = 0;
  if (!CheckCuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount") ||
      device_count == 0) {
    std::cerr << "No CUDA-capable GPU is available in this environment.\n";
    return 1;
  }

  cudaDeviceProp properties{};
  if (!CheckCuda(cudaGetDeviceProperties(&properties, 0),
                 "cudaGetDeviceProperties")) {
    return 1;
  }

  std::cout << "GPU: " << properties.name << '\n'
            << "Compute capability: " << properties.major << '.'
            << properties.minor << '\n'
            << "Input: " << options.input_path << '\n'
            << "Output: " << options.output_path << '\n'
            << "Scale: " << options.scale << '\n'
            << "Input dimensions: " << source.width << " x " << source.height
            << '\n'
            << "Output dimensions: " << destination.width << " x "
            << destination.height << '\n';

  Npp8u* device_source = nullptr;
  Npp8u* device_destination = nullptr;
  const std::size_t source_bytes = source.pixels.size() * sizeof(Npp8u);
  const std::size_t destination_bytes =
      destination.pixels.size() * sizeof(Npp8u);

  if (!CheckCuda(cudaMalloc(reinterpret_cast<void**>(&device_source),
                            source_bytes),
                 "cudaMalloc source") ||
      !CheckCuda(cudaMalloc(reinterpret_cast<void**>(&device_destination),
                            destination_bytes),
                 "cudaMalloc destination")) {
    if (device_source != nullptr) cudaFree(device_source);
    if (device_destination != nullptr) cudaFree(device_destination);
    return 1;
  }

  bool success = CheckCuda(
      cudaMemcpy(device_source, source.pixels.data(), source_bytes,
                 cudaMemcpyHostToDevice),
      "Host-to-device copy");

  cudaEvent_t start_event{};
  cudaEvent_t stop_event{};
  bool events_created = false;
  if (success) {
    success = CheckCuda(cudaEventCreate(&start_event), "cudaEventCreate start") &&
              CheckCuda(cudaEventCreate(&stop_event), "cudaEventCreate stop");
    events_created = success;
  }

  if (success) {
    const NppiSize source_size{source.width, source.height};
    const NppiRect source_roi{0, 0, source.width, source.height};
    const NppiSize destination_size{destination.width, destination.height};
    const NppiRect destination_roi{0, 0, destination.width, destination.height};

    success = CheckCuda(cudaEventRecord(start_event, 0), "cudaEventRecord start");
    if (success) {
      success = CheckNpp(
          nppiResize_8u_C1R(
              device_source, source.width, source_size, source_roi,
              device_destination, destination.width, destination_size,
              destination_roi, NPPI_INTER_LINEAR),
          "nppiResize_8u_C1R");
    }
    if (success) {
      success = CheckCuda(cudaEventRecord(stop_event, 0), "cudaEventRecord stop") &&
                CheckCuda(cudaEventSynchronize(stop_event),
                          "cudaEventSynchronize");
    }
    if (success) {
      float elapsed_ms = 0.0F;
      success = CheckCuda(cudaEventElapsedTime(&elapsed_ms, start_event,
                                               stop_event),
                          "cudaEventElapsedTime");
      if (success) std::cout << "GPU resize time (ms): " << elapsed_ms << '\n';
    }
  }

  if (events_created) {
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
  }

  if (success) {
    success = CheckCuda(
        cudaMemcpy(destination.pixels.data(), device_destination,
                   destination_bytes, cudaMemcpyDeviceToHost),
        "Device-to-host copy");
  }

  cudaFree(device_source);
  cudaFree(device_destination);

  if (!success) return 1;
  if (!WritePgm(options.output_path, destination, &error)) {
    std::cerr << "Output error: " << error << '\n';
    return 1;
  }

  std::cout << "Status: SUCCESS\nSaved output: " << options.output_path << '\n';
  return 0;
}
