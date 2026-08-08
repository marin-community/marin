MOK_ROOT ?=
THUNDERKITTENS_ROOT ?= $(MOK_ROOT)/third_party/ThunderKittens
NVCC ?= nvcc
PYTHON ?= python3
OUT ?= _mok_gmm_probe$(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
MOK_HEADERS := $(wildcard $(MOK_ROOT)/csrc/*.cuh)
THUNDERKITTENS_HEADERS := $(shell find $(THUNDERKITTENS_ROOT)/include -type f -name '*.cuh' 2>/dev/null)

PYTHON_INCLUDES := $(shell $(PYTHON) -c "import sysconfig; print('-I' + sysconfig.get_path('include'))")
PYTORCH_INCLUDES := $(shell $(PYTHON) -c "from torch.utils.cpp_extension import include_paths; print(' '.join('-I' + p for p in include_paths()))")
PYTORCH_LIBDIR := $(shell $(PYTHON) -c "from torch.utils.cpp_extension import library_paths; print(' '.join('-L' + p for p in library_paths()))")

NVCCFLAGS := -DNDEBUG -lineinfo
NVCCFLAGS += --expt-extended-lambda --expt-relaxed-constexpr
NVCCFLAGS += -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing
NVCCFLAGS += -forward-unknown-to-host-compiler -ftemplate-backtrace-limit=0
NVCCFLAGS += -std=c++20 -O3 --use_fast_math
NVCCFLAGS += -Xptxas=--verbose -Xptxas=--warn-on-spills
NVCCFLAGS += -I$(MOK_ROOT)/csrc -I$(THUNDERKITTENS_ROOT)/include
NVCCFLAGS += -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__
NVCCFLAGS += -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__
NVCCFLAGS += -DKITTENS_SM100 -gencode arch=compute_100a,code=sm_100a
NVCCFLAGS += -DTORCH_API_INCLUDE_EXTENSION_H -DTORCH_EXTENSION_NAME=_mok_gmm_probe
NVCCFLAGS += $(PYTHON_INCLUDES) $(PYTORCH_INCLUDES) $(PYTORCH_LIBDIR)
NVCCFLAGS += -shared -fPIC -lrt -lpthread -ldl -lcuda -lcudadevrt -lcudart_static
NVCCFLAGS += -ltorch_python -ltorch_cuda -ltorch_cpu -ltorch -lc10_cuda -lc10
NVCCFLAGS += -diag-suppress 3189

all: $(OUT)

$(OUT): mok_gmm_probe.cu generated_map_fold.inc $(MOK_HEADERS) $(THUNDERKITTENS_HEADERS)
	@test -f "$(MOK_ROOT)/csrc/mok_megakernel.cuh" || (echo "MOK_ROOT must name a Mixture-of-Kittens checkout" >&2; exit 1)
	@test -f "$(THUNDERKITTENS_ROOT)/include/kittens.cuh" || (echo "ThunderKittens submodule is missing" >&2; exit 1)
	@mkdir -p "$(dir $(OUT))"
	$(NVCC) $< $(NVCCFLAGS) -o "$@"

.PHONY: all
