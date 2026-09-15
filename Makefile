NVCC      := nvcc
ARCH      ?= sm_89
CXXSTD    := c++20
NVFLAGS   := -std=$(CXXSTD) -O3 -arch=$(ARCH) $(EXTRA_NVFLAGS)
PTXAS_FLAGS := -Xptxas -v
INCLUDES  := -I.
BIN_DIR   := bin

CXX       := g++
CXXFLAGS  := -std=$(CXXSTD) -O3 -fopenmp -march=native

AP_HEADERS := \
	include/common.hpp \
	include/distance.hpp \
	include/data_io.hpp \
	include/qg.hpp \
	include/metric.hpp \
	include/quant.hpp \
	include/utils.cuh \
	src/adaptive_search.cuh \
	src/adaptive_search_utils.cuh \
	src/adaptive_search_config.cuh

BUILD_HEADERS := \
	include/common.hpp \
	include/distance.hpp \
	include/data_io.hpp \
	include/encode.hpp \
	include/metric.hpp \
	include/pack.hpp \
	include/quant.hpp \
	include/rotator.hpp \
	include/sketch.hpp

.PHONY: all clean

all: gpu_flashgann buildindex

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

gpu_flashgann: main.cu gpu_search_adaptive.cu $(AP_HEADERS) | $(BIN_DIR)
	$(NVCC) $(NVFLAGS) $(PTXAS_FLAGS) $(INCLUDES) -o $@ main.cu gpu_search_adaptive.cu
	mv $@ $(BIN_DIR)/

buildindex: tools/buildindex.cc $(BUILD_HEADERS) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ tools/buildindex.cc
	mv $@ $(BIN_DIR)/

clean:
	rm -f gpu_flashgann buildindex $(BIN_DIR)/gpu_flashgann $(BIN_DIR)/buildindex *.o
