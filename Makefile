CXX ?= gcc
NVCC ?= nvcc

ifeq ($(PYTHON_VERSION), 2)
$(warning using python version 2)
PYTHON ?= python2
else
$(warning default using python version 3)
PYTHON ?= python3
endif

ifeq ($(GPU_MODE), true)
$(warning dynamic_embedding_server is building with GPU enabled)
CUDA_HOME ?= /usr/local/cuda
CUDA_CFLAGS ?= \
	-DGOOGLE_CUDA=1 \
	-I$(CUDA_HOME)/include

CUDA_LDFLAGS ?= \
	-L$(CUDA_HOME)/lib64 \
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
	-Iinclude \
	-I.

LIBNAME := dynamic_embedding_server

CXX_CFLAGS := -std=c++11 \
	-fstack-protector \
	-Wall \
	-Werror \
	-Wno-sign-compare \
	-Wformat \
	-Wformat-security

LDFLAGS := -shared \
	-fstack-protector \
	-fpic \
	$(CUDA_LDFLAGS)

GAZER_LIB := gazer/libgazer.so
include gazer/cc/Makefile

.PHONY: gazer
gazer: $(GAZER_LIB)
	@echo -e "\033[1;33m[BUILD] $$t \033[0m" \
        "build wheel package"
	cd gazer/; $(PYTHON) setup.py bdist_wheel
	@ls gazer/dist/*.whl

DES_LIB := $(LIBNAME)/lib$(LIBNAME).so
include $(LIBNAME)/cc/Makefile

.PHONY: des
des: $(DES_LIB)
	@echo -e "\033[1;33m[BUILD] $$t \033[0m" \
        "build wheel package"
	cd $(LIBNAME); $(PYTHON) setup.py bdist_wheel
	@ls $(LIBNAME)/dist/*.whl

.PHONY: clean
clean:
	@rm -fr gazer/dist/
	@rm -fr gazer/build/
	@rm -fr gazer/*.egg-info/
	@rm -fr $(LIBNAME)/dist/
	@rm -fr $(LIBNAME)/build/
	@rm -fr $(LIBNAME)/*.egg-info/
	@rm -fr third_party/rapidjson/build
	@rm -fr third_party/grpc/build
	@rm -fr third_party/protobuf/build
	@echo "remove .o/.d/.so/.pb*"
	@find ./ -name *.pb.* -exec rm -rf {} \;
	@find ./ -name *_pb2* -exec rm -fr {} \;

.DEFAULT_GOAL := gazer
