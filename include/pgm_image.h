#ifndef CUDA_IMAGE_RESIZING_PGM_IMAGE_H_
#define CUDA_IMAGE_RESIZING_PGM_IMAGE_H_

#include <cstdint>
#include <string>
#include <vector>

struct PgmImage {
  int width = 0;
  int height = 0;
  std::vector<std::uint8_t> pixels;
};

bool ReadPgm(const std::string& path, PgmImage* image, std::string* error);
bool WritePgm(const std::string& path, const PgmImage& image,
              std::string* error);

#endif  // CUDA_IMAGE_RESIZING_PGM_IMAGE_H_
