SRC_DIR := gazer/cc
THIRD_PARTY_DIR := third_party
GLIBCXX_USE_CXX11_ABI := 0

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

$(warning "Tensorflow Version" $(TENSORFLOW_VERSION))
$(warning "CFLAGS" $(TENSORFLOW_CFLAGS))
$(warning "LDFLAGS" $(TENSORFLOW_LDFLAGS))
$(warning "GLIBCXX_USE_CXX11_ABI" $(GLIBCXX_USE_CXX11_ABI))

CC_SOURCES := $(shell \
        find $(SRC_DIR)/client $(SRC_DIR)/graph $(SRC_DIR)/ops $(SRC_DIR)/util -type f \
        \( -name "*.cc" ! -name "*.cu*" \) \
        -exec realpath {} --relative-to . \;)

# rapidjson
RAPIDJSON_DIR := $(THIRD_PARTY_DIR)/rapidjson
RAPIDJSON_INCLUDE := $(RAPIDJSON_DIR)/build/include
RAPIDJSON_LIB := $(RAPIDJSON_DIR)/build/lib
rapidjson:
	@echo "prepare rapidjson library ..."
	@if [ ! -d "${RAPIDJSON_DIR}/build" ]; then cd "${RAPIDJSON_DIR}"; bash ./build.sh; fi
	@echo "rapidjson done"

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
PROTOC_GRPC_PLUGIN := $(GRPC_DIR)/build/bin/grpc_cpp_plugin
grpc: protobuf
	@echo "prepare grpc library ..."
	@if [ ! -d "${GRPC_DIR}/build" ]; then cd "${GRPC_DIR}"; TF_VERSION=${TF_VERSION} bash ./build.sh; fi
	@echo "grpc done"

PROTO_DIR := $(SRC_DIR)/proto
PROTO_BUILT_DIR := $(PROTO_DIR)
PROTO_OBJ := $(PROTO_BUILT_DIR)/scheduler.pb.o $(PROTO_BUILT_DIR)/scheduler.grpc.pb.o

proto: grpc
	mkdir -p $(dir $@)
	@echo "generating pb file"
	$(PROTOC) --cpp_out=./ gazer/cc/proto/scheduler.proto
	$(PROTOC) --grpc_out=./ --plugin=protoc-gen-grpc=$(PROTOC_GRPC_PLUGIN) gazer/cc/proto/scheduler.proto
	$(CXX) $(GLIBCXX_USE_CXX11_ABI) -std=c++11 -fpic -I. -I$(PROTOBUF_INCLUDE) -I$(GRPC_INCLUDE) -c $(PROTO_DIR)/scheduler.pb.cc -o $(PROTO_BUILT_DIR)/scheduler.pb.o
	$(CXX) $(GLIBCXX_USE_CXX11_ABI) -std=c++11 -fpic -I. -I$(PROTOBUF_INCLUDE) -I$(GRPC_INCLUDE) -c $(PROTO_DIR)/scheduler.grpc.pb.cc -o $(PROTO_BUILT_DIR)/scheduler.grpc.pb.o

OBJS := $(CC_SOURCES:.cc=.o)
DEPS := $(OBJS:.o=.d)

CXX_CFLAGS += -I$(PROTOBUF_INCLUDE)
CXX_CFLAGS += -I$(GRPC_INCLUDE)
CXX_CFLAGS += -I$(RAPIDJSON_INCLUDE)

LDFLAGS += -L$(PROTOBUF_LIB)
LDFLAGS += -L$(GRPC_LIB)
LDFLAGS += -lprotobuf
LDFLAGS += -lgrpc++
LDFLAGS += -lgrpc
LDFLAGS += -lssl
LDFLAGS += -lcrypto
LDFLAGS += -lcares

$(OBJS): %.o:%.cc protobuf grpc proto rapidjson
	mkdir -p $(dir $@)
	$(CXX) $(CFLAGS) $(CXX_CFLAGS) $(TENSORFLOW_CFLAGS)\
        -MP -MMD -MF $<.d -o $@ -c $< -fpic

-include $(DEPS)

$(GAZER_LIB): $(OBJS) 
	mkdir -p $(dir $@)
	$(CXX) $(CFLAGS) -std=c++11 -Wl,--version-script=gazer/gazer.script \
        -o $@ $(OBJS) $(PROTO_OBJ) \
        $(TENSORFLOW_LDFLAGS) \
		$(LDFLAGS)

