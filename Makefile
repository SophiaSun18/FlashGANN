NVCC      := nvcc
ARCH      ?= sm_89
CXXSTD    := c++20
NVFLAGS   := -std=$(CXXSTD) -O3 -arch=$(ARCH) $(EXTRA_NVFLAGS)
PTXAS_FLAGS := -Xptxas -v
INCLUDES  := -I.
BIN_DIR   := bin

AP_HEADERS := \
	include/common.hpp \
	include/distance.hpp \
	include/data_io.hpp \
	include/qg.hpp \
	include/metric.hpp \
	include/utils.cuh \
	src/adaptive_search.cuh \
	src/adaptive_search_utils.cuh \
	src/adaptive_search_config.cuh

.PHONY: all clean

all: gpu_flashgann

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

gpu_flashgann: main.cu gpu_search_adaptive.cu $(AP_HEADERS) | $(BIN_DIR)
	$(NVCC) $(NVFLAGS) $(PTXAS_FLAGS) $(INCLUDES) -o $@ main.cu gpu_search_adaptive.cu
	mv $@ $(BIN_DIR)/

clean:
	rm -f gpu_flashgann $(BIN_DIR)/gpu_flashgann *.o
