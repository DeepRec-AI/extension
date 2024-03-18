CXX ?= gcc
NVCC ?= nvcc
GLIBCXX_USE_CXX11_ABI := 0
THIRD_PARTY_DIR := third_party

ifeq ($(PYTHON_VERSION), 2)
$(warning using python version 2)
PYTHON ?= python2
else
$(warning default using python version 3)
PYTHON ?= python3
endif

TENSORFLOW_VERSION := $(shell \
        LD_LIBRARY_PATH=/usr/local/gcc-5.3.0/lib64 $(PYTHON) -c \
        "import tensorflow as tf; print(tf.__version__)" 2>/dev/null)

ifeq ($(TENSORFLOW_VERSION), 1.12.2-PAI2209u1)
TF_VERSION ?= 112
else
TF_VERSION ?= 115
endif
TENSORFLOW_CFLAGS := $(shell \
        $(PYTHON) -c \
        "import tensorflow as tf; cflags=tf.sysconfig.get_compile_flags(); print(' '.join([c.replace('-I', '-isystem ', 1) + ' ' + c.replace('-I', '-isystem ', 1) + '/external/sparsehash_c11' + ' ' + c.replace('-I', '-isystem ', 1) + '/external/com_github_google_leveldb/include/' + ' ' + c.replace('-I', '-isystem ', 1) + '/external/libcuckoo/' if c.startswith('-I') else c for c in cflags]))" 2>/dev/null)

GLIBCXX_USE_CXX11_ABI := $(shell \
        $(PYTHON) -c \
        "import tensorflow as tf; cflags=tf.sysconfig.get_compile_flags(); print(' '.join([c if c.find('GLIBCXX_USE_CXX11_ABI') != -1 else '' for c in cflags]))" 2>/dev/null)

TENSORFLOW_LDFLAGS := \
        -Wl,-rpath='$$ORIGIN/..:$$ORIGIN/../tensorflow' \
        $(shell \
        $(PYTHON) -c \
        "import tensorflow as tf; ldflags=tf.sysconfig.get_link_flags(); print(' '.join(ldflags))" 2>/dev/null)

$(warning "Tensorflow Version: " $(TENSORFLOW_VERSION))
$(warning "Tensorflow CFLAGS: " $(TENSORFLOW_CFLAGS))
$(warning "Tensorflow LDFLAGS: " $(TENSORFLOW_LDFLAGS))

ifeq ($(GPU_MODE), true)
$(warning dynamic_embedding_server is building with GPU enabled)
CUDA_HOME ?= /usr/local/cuda
CUDA_CFLAGS ?= -DGOOGLE_CUDA=1 \
	-I$(CUDA_HOME)/include \
	-DEIGEN_MPL2_ONLY \
	-DEIGEN_MAX_ALIGN_BYTES=64 \
	-DEIGEN_HAS_TYPE_TRAITS=0
CUDA_LDFLAGS ?= -L$(CUDA_HOME)/lib64 \
	-lcudart \
	-L/usr/lib64 \
	-lnccl
else
CUDA_HOME := 
CUDA_CFLAGS :=
CUDA_LDFLAGS :=
endif

CFLAGS := -O3 -g \
	-DNDEBUG \
	$(CUDA_CFLAGS) \
	-I.

CXX_CFLAGS := -std=c++11 \
	-fstack-protector \
	-Wall \
	-Wno-sign-compare \
	-Wformat \
	-Wformat-security

LDFLAGS := -shared \
	-fstack-protector \
	-fpic \
	$(CUDA_LDFLAGS)

# protobuf
PROTOBUF_DIR := $(THIRD_PARTY_DIR)/protobuf
PROTOBUF_INCLUDE := $(PROTOBUF_DIR)/build/include
PROTOBUF_LIB := $(PROTOBUF_DIR)/build/lib
PROTOC := $(PROTOBUF_DIR)/build/bin/protoc
protobuf:
	@echo "prepare protobuf library ..."
	@if [ ! -d "${PROTOBUF_DIR}/build" ]; then cd "${PROTOBUF_DIR}"; TF_VERSION=${TF_VERSION} bash ./build.sh; fi
	@echo "protobuf done"

# grpc
GRPC_DIR := $(THIRD_PARTY_DIR)/grpc
GRPC_INCLUDE := $(GRPC_DIR)/build/include
GRPC_LIB := $(GRPC_DIR)/build/lib
PROTOC_GRPC_CPP_PLUGIN := $(GRPC_DIR)/build/bin/grpc_cpp_plugin
PROTOC_GRPC_PYTHON_PLUGIN := $(GRPC_DIR)/build/bin/grpc_python_plugin
grpc: protobuf
	@echo "prepare grpc library ..."
	@if [ ! -d "${GRPC_DIR}/build" ]; then cd "${GRPC_DIR}"; TF_VERSION=${TF_VERSION} bash ./build.sh; fi
	@echo "grpc done"

# rapidjson
RAPIDJSON_DIR := $(THIRD_PARTY_DIR)/rapidjson
RAPIDJSON_INCLUDE := $(RAPIDJSON_DIR)/build/include
RAPIDJSON_LIB := $(RAPIDJSON_DIR)/build/lib
rapidjson:
	@echo "prepare rapidjson library ..."
	@if [ ! -d "${RAPIDJSON_DIR}/build" ]; then cd "${RAPIDJSON_DIR}"; bash ./build.sh; fi
	@echo "rapidjson done"

# gazer
GAZER := gazer
GAZER_LIB := $(GAZER)/$(GAZER)/lib$(GAZER).so
include $(GAZER)/$(GAZER)/cc/Makefile

.PHONY: gazer
gazer: $(GAZER_LIB)
	@echo -e "\033[1;33m[BUILD] $$t \033[0m" \
        "build wheel package"
	cd $(GAZER)/; $(PYTHON) setup.py bdist_wheel
	@ls $(GAZER)/dist/*.whl

# dynamic_embedding_server
DES := dynamic_embedding_server
DES_LIB := $(DES)/$(DES)/lib$(DES).so
include $(DES)/$(DES)/cc/Makefile

.PHONY: des
des: $(DES_LIB)
	@echo -e "\033[1;33m[BUILD] $$t \033[0m" \
        "build wheel package"
	cd $(DES); $(PYTHON) setup.py bdist_wheel
	@ls $(DES)/dist/*.whl

# tf_fault_tolerance
TFT := tf_fault_tolerance
include $(TFT)/Makefile

.PHONY: tft
tft: $(TFT_LIB)
	@echo -e "\033[1;33m[BUILD] $$t \033[0m" \
        "build wheel package"
	cd $(TFT); $(PYTHON) setup.py bdist_wheel
	@ls $(TFT)/dist/*.whl

.PHONY: all
all: gazer des tft

ALL_MOUDLES := $(GAZER) $(DES) $(TFT)

.PHONY: clean
clean:
	@rm -fr $(addsuffix /dist/, $(ALL_MOUDLES))
	@rm -fr $(addsuffix /build/, $(ALL_MOUDLES))
	@rm -fr $(addsuffix /*.egg-info/, $(ALL_MOUDLES))
	@rm -fr third_party/rapidjson/build
	@rm -fr third_party/grpc/build
	@rm -fr third_party/protobuf/build
	@echo "remove .o/.d/.so/.pb*"
	@find $(ALL_MOUDLES) -name *.o -exec rm -fr {} \;
	@find $(ALL_MOUDLES) -name *.d -exec rm -fr {} \;
	@find $(ALL_MOUDLES) -name *.so -exec rm -fr {} \;
	@find $(ALL_MOUDLES) -name *.pb.* -exec rm -rf {} \;
	@find $(ALL_MOUDLES) -name *_pb2* -exec rm -fr {} \;

.DEFAULT_GOAL := gazer
