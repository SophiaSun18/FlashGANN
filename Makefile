NVCC      := nvcc
ARCH      ?= sm_89
CXXSTD    := c++20
NVFLAGS   := -std=$(CXXSTD) -O3 -arch=$(ARCH) -Xcompiler -fopenmp $(EXTRA_NVFLAGS)
NVLIBS    := -lgomp
PTXAS_FLAGS := -Xptxas -v
INCLUDES  := -I.
BIN_DIR   := bin

# gpu_cagra builds against the standalone cuVS 26.08 tree with CUDA 13.1 and its conda host toolchain
CAGRA_NVCC  ?= /usr/local/cuda-13.1/bin/nvcc
CAGRA_ENV   ?= /home/sunzeai/anaconda3/envs/cuvs
CUVS_ROOT   ?= /home/sunzeai/proj/cuvs_official_20260602/cpp
CUVS_BUILD  := $(CUVS_ROOT)/build
CAGRA_FLAGS := -std=$(CXXSTD) -O3 -arch=$(ARCH) -ccbin $(CAGRA_ENV)/bin/x86_64-conda-linux-gnu-c++ \
	-Xcompiler -fopenmp --expt-extended-lambda --expt-relaxed-constexpr \
	-DLIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE -DRAFT_SYSTEM_LITTLE_ENDIAN=1 \
	-DRAFT_LOG_ACTIVE_LEVEL=RAPIDS_LOGGER_LOG_LEVEL_INFO \
	-DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP
CAGRA_INCLUDES := -I. \
	-I$(CUVS_BUILD)/_deps/cccl-src/thrust \
	-I$(CUVS_BUILD)/_deps/cccl-src/libcudacxx/include \
	-I$(CUVS_BUILD)/_deps/cccl-src/cub \
	-isystem $(CUVS_BUILD)/_deps/rmm-src/cpp/include \
	-isystem $(CUVS_BUILD)/_deps/rmm-build/include \
	-isystem $(CAGRA_ENV)/targets/x86_64-linux/include \
	-isystem $(CAGRA_ENV)/targets/x86_64-linux/include/cccl \
	-isystem $(CUVS_BUILD)/_deps/raft-src/cpp/include \
	-isystem $(CUVS_BUILD)/_deps/raft-build/include \
	-isystem $(CUVS_ROOT)/include \
	-isystem $(CUVS_BUILD)/include
CAGRA_LIBS := -L$(CUVS_BUILD) -L$(CUVS_BUILD)/_deps/rmm-build -L$(CAGRA_ENV)/lib \
	-lcuvs -lrmm -lrapids_logger -lgomp \
	-Xlinker -rpath=$(CUVS_BUILD):$(CUVS_BUILD)/_deps/rmm-build:$(CAGRA_ENV)/lib

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
	include/hash_table.cuh \
	include/distance.cuh \
	include/beam_management.cuh \
	include/quant.cuh \
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

all: gpu_flashgann gpu_cagra gpu_pathw gpu_rabitq buildindex

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

gpu_flashgann: main.cu gpu_search_adaptive.cu $(AP_HEADERS) | $(BIN_DIR)
	$(NVCC) $(NVFLAGS) $(PTXAS_FLAGS) $(INCLUDES) -DGPU_SEARCH_MODE=1 -o $@ main.cu gpu_search_adaptive.cu $(NVLIBS)
	mv $@ $(BIN_DIR)/

gpu_cagra: main.cu gpu_search_cagra.cu include/index.hpp include/common.hpp include/data_io.hpp include/distance.hpp include/metric.hpp | $(BIN_DIR)
	$(CAGRA_NVCC) $(CAGRA_FLAGS) $(CAGRA_INCLUDES) -DGPU_SEARCH_MODE=2 -o $@ main.cu gpu_search_cagra.cu $(CAGRA_LIBS)
	mv $@ $(BIN_DIR)/

gpu_pathw: main.cu gpu_search_pathw.cu include/index.hpp src/pathw_search.cuh src/pathw_utils.cuh include/utils.cuh include/hash_table.cuh include/beam_management.cuh include/common.hpp include/data_io.hpp include/distance.hpp include/metric.hpp include/quant.hpp | $(BIN_DIR)
	$(NVCC) $(NVFLAGS) $(PTXAS_FLAGS) $(INCLUDES) -DGPU_SEARCH_MODE=3 -o $@ main.cu gpu_search_pathw.cu $(NVLIBS)
	mv $@ $(BIN_DIR)/

gpu_rabitq: main.cu gpu_search_rabitq.cu include/common.hpp include/distance.hpp include/data_io.hpp include/qg.hpp include/metric.hpp include/quant.hpp include/utils.cuh include/hash_table.cuh include/distance.cuh include/beam_management.cuh include/quant.cuh src/adaptive_search_config.cuh src/rabitq_utils.cuh src/rabitq_search.cuh | $(BIN_DIR)
	$(NVCC) $(NVFLAGS) $(PTXAS_FLAGS) $(INCLUDES) -DGPU_SEARCH_MODE=4 -o $@ main.cu gpu_search_rabitq.cu $(NVLIBS)
	mv $@ $(BIN_DIR)/

buildindex: tools/buildindex.cc $(BUILD_HEADERS) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ tools/buildindex.cc
	mv $@ $(BIN_DIR)/

clean:
	rm -f gpu_flashgann gpu_cagra gpu_pathw gpu_rabitq buildindex $(BIN_DIR)/gpu_flashgann $(BIN_DIR)/gpu_cagra $(BIN_DIR)/gpu_pathw $(BIN_DIR)/gpu_rabitq $(BIN_DIR)/buildindex *.o
