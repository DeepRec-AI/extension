SRC_DIR := dynamic_embedding_server/cc/
THIRD_PARTY_DIR := third_party

TENSORFLOW_VERSION := $(shell $(PYTHON) -c \
	    "import tensorflow as tf; print(tf.__version__)" 2>/dev/null)

ifeq ($(TENSORFLOW_VERSION), 1.4.0-PAI1807)
TENSORFLOW_CFLAGS := $(shell \
		     LD_LIBRARY_PATH=/usr/local/lib64/ $(PYTHON) -c \
        "import tensorflow as tf; include=tf.sysconfig.get_include(); print(' '.join(['-D_GLIBCXX_USE_CXX11_ABI=0','-I' + include + ' -I' + include + '/external/sparsehash_c11' + ' -I' + include + '/external/com_github_google_leveldb/include/' + ' -I' + include + '/external/libcuckoo/' + ' -I' + include + '/external/nsync/public/' + ' -I' + include + '/external/com_google_absl' + ' -I' + include + '/external/grpc']))" 2>/dev/null)
TENSORFLOW_LDFLAGS := \
        -Wl,-rpath='$$ORIGIN/..:$$ORIGIN/../tensorflow' \
        $(shell \
	LD_LIBRARY_PATH=/usr/local/lib64/ $(PYTHON) -c \
        "import tensorflow as tf; lib=tf.sysconfig.get_lib(); print('-ltensorflow_framework -L' + lib)" 2>/dev/null)
else
TENSORFLOW_CFLAGS := $(shell \
        $(PYTHON) -c \
        "import tensorflow as tf; cflags=tf.sysconfig.get_compile_flags(); print(' '.join([c.replace('-I', '-isystem ', 1) + ' ' + c.replace('-I', '-isystem ', 1) + '/external/sparsehash_c11' + ' ' + c.replace('-I', '-isystem ', 1) + '/external/com_github_google_leveldb/include/' + ' ' + c.replace('-I', '-isystem ', 1) + '/external/libcuckoo/' + ' ' + c.replace('-I', '-isystem ', 1) + '/external/grpc/' if c.startswith('-I') else c for c in cflags]))" 2>/dev/null)
TENSORFLOW_LDFLAGS := \
        -Wl,-rpath='$$ORIGIN/..:$$ORIGIN/../tensorflow' \
        $(shell \
        $(PYTHON) -c \
	"import tensorflow as tf; ldflags=tf.sysconfig.get_link_flags(); lib=tf.sysconfig.get_lib(); print(' '.join(ldflags))" 2>/dev/null)
endif


$(warning "Tensorflow Version" $(TENSORFLOW_VERSION))
$(warning "CFLAGS" $(TENSORFLOW_CFLAGS))
$(warning "LDFLAGS" $(TENSORFLOW_LDFLAGS))

# protobuf
PROTOBUF_DIR := $(THIRD_PARTY_DIR)/protobuf
PROTOBUF_INCLUDE := $(PROTOBUF_DIR)/build/include
PROTOBUF_LIB := $(PROTOBUF_DIR)/build/lib
PROTOC := $(PROTOBUF_DIR)/build/bin/protoc
protobuf:
	@echo "prepare protobuf library ..."
	@if [ ! -d "${PROTOBUF_DIR}/build" ]; then cd "${PROTOBUF_DIR}"; bash ./build.sh; fi
	@echo "protobuf done"

# grpc
GRPC_DIR := $(THIRD_PARTY_DIR)/grpc
GRPC_INCLUDE := $(GRPC_DIR)/build/include
GRPC_LIB := $(GRPC_DIR)/build/lib
PROTOC_GRPC_PLUGIN := $(GRPC_DIR)/build/bin/grpc_cpp_plugin
grpc: protobuf
	@echo "prepare grpc library ..."
	@if [ ! -d "${GRPC_DIR}/build" ]; then cd "${GRPC_DIR}"; bash ./build.sh; fi
	@echo "grpc done"

PROTO_DIR := dynamic_embedding_server/proto
PROTO_OBJ := $(PROTO_BUILT_DIR)/elastic_training.pb.o $(PROTO_BUILT_DIR)/elastic_training.grpc.pb.o 

proto: grpc
	mkdir -p $(dir $@)
	@echo "generating pb file"
	$(PROTOC) --cpp_out=./ dynamic_embedding_server/proto/elastic_training.proto
	$(PROTOC) --grpc_out=./ --plugin=protoc-gen-grpc=$(PROTOC_GRPC_PLUGIN) dynamic_embedding_server/proto/elastic_training.proto
	$(PROTOC) --proto_path=. --python_out=./ dynamic_embedding_server/proto/elastic_training.proto
	$(CXX) -std=c++11 -fPIC -I./ -I$(PROTOBUF_INCLUDE) -I$(GRPC_INCLUDE) -c $(PROTO_DIR)/elastic_training.pb.cc -o $(PROTO_BUILT_DIR)/elastic_training.pb.o
	$(CXX) -std=c++11 -fPIC -I./ -I$(PROTOBUF_INCLUDE) -I$(GRPC_INCLUDE) -c $(PROTO_DIR)/elastic_training.grpc.pb.cc -o $(PROTO_BUILT_DIR)/elastic_training.grpc.pb.o

CC_SOURCES := $(shell \
        find $(SRC_DIR) -type f \
        \( -name "*.cc" ! -name "*.cu*" \) \
        -exec realpath {} --relative-to . \;)

OBJS := $(CC_SOURCES:.cc=.o)
DEPS := $(OBJS:.o=.d)

CXX_CFLAGS += -I$(PROTOBUF_INCLUDE)
CXX_CFLAGS += -I$(GRPC_INCLUDE)

LDFLAGS += -L$(PROTOBUF_LIB)
LDFLAGS += -L$(GRPC_LIB)
LDFLAGS += -lprotobuf
LDFLAGS += -lssl
LDFLAGS += -lcrypto


$(OBJS): %.o:%.cc protobuf grpc proto
	mkdir -p $(dir $@)
	$(CXX) $(CFLAGS) $(TENSORFLOW_CFLAGS) $(CXX_CFLAGS) \
        -MP -MMD -MF $<.d -o $@ -c $< -fpic
	sed -i '/site-packages/d' $<.d
	sed -i '/^$$/N;/^\n$$/D' $<.d

CU_SOURCES = $(shell \
	find -type f \
	\( -name '*.cu.cc' ! -name "*test*" \) \
	-exec realpath {} --relative-to . \;')

$(CU_OBJS): %.o:%.cc
	mkdir -p $(dir $@)
	@$(NVCC) -M $< $(CFLAGS) -x cu \
	 | grep -v '/usr/' \
	 | sed 's|$(notdir $@)|$@|g' \
	 | sed 's|\./||g' \
	 > $<.d
	$(NVCC) \
		--std=c++11 \
		--expt-relaxed-constexpr \
		--expt-extended-lambda \
		--disable-warnings \
		-o $@ -c $< $(CFLAGS) -x cu \
		-Xcompiler -fPIC

-include $(DEPS)

ifneq ($(CUDA_CFLAGS),)
$(DES_LIB): $(OBJS) $(CU_OBJS)
	mkdir -p $(dir $@)
	$(CXX) $(CFLAGS) -std=c++11 \
        -o $@ $(PROTO_OBJ) $(OBJS)  $(CU_OBJS) \
        $(TENSORFLOW_LDFLAGS) $(LDFLAGS) $(CUDA_LDFLAGS)
else
$(DES_LIB): $(OBJS)
	mkdir -p $(dir $@)
	$(CXX) $(CFLAGS) -std=c++11 \
        -o $@ $(PROTO_OBJ) $(OBJS) \
        $(TENSORFLOW_LDFLAGS) $(LDFLAGS)
endif
        
