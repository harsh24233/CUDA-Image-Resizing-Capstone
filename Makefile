CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(shell which nvcc 2>/dev/null || echo $(CUDA_HOME)/bin/nvcc)
TARGET := bin/image_resize
SRC := src/image_resize.cu

# NPP image geometry operations are commonly provided by nppig.
# If your installed CUDA version uses different library packaging, adjust
# NPP_LIBS after checking the libraries installed in your environment.
NPP_LIBS ?= -lnppig -lnppc

CUDA_INC := $(wildcard $(CUDA_HOME)/include)
CUDA_LIB := $(wildcard $(CUDA_HOME)/lib64)

NVCCFLAGS ?= -O2 -std=c++14 -Iinclude $(if $(CUDA_INC),-I$(CUDA_INC),)
LDFLAGS ?= $(if $(CUDA_LIB),-L$(CUDA_LIB),)
LDLIBS := $(NPP_LIBS) -lcudart

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC) include/pgm_image.h
	@mkdir -p bin
	$(NVCC) $(NVCCFLAGS) $(SRC) -o $@ $(LDFLAGS) $(LDLIBS)

clean:
	rm -rf bin
