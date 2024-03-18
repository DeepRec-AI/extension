DES_SRC_DIR := $(DES)/$(DES)/cc/
DES_PROTO_DIR := $(DES)/$(DES)/proto
DES_PROTO_OBJ := $(DES_PROTO_DIR)/elastic_training.pb.o $(DES_PROTO_DIR)/elastic_training.grpc.pb.o

DES_CFLAGS := $(CFLAGS)
DES_CFLAGS += -I$(PROTOBUF_INCLUDE)
DES_CFLAGS += -I$(GRPC_INCLUDE)
DES_CFLAGS += -I$(RAPIDJSON_INCLUDE)
DES_CFLAGS += -I$(DES)

DES_LDFLAGS := $(LDFLAGS)
DES_LDFLAGS += -L$(PROTOBUF_LIB)
DES_LDFLAGS += -L$(GRPC_LIB)
DES_LDFLAGS += -lprotobuf
DES_LDFLAGS += -lgrpc++
DES_LDFLAGS += -lgrpc
DES_LDFLAGS += -lssl
DES_LDFLAGS += -lcrypto

des_proto: grpc
	mkdir -p $(dir $@)
	@echo "generating pb file"
	$(PROTOC) --cpp_out=./ $(DES_PROTO_DIR)/elastic_training.proto
	$(PROTOC) --grpc_out=./ --plugin=protoc-gen-grpc=$(PROTOC_GRPC_PLUGIN) $(DES_PROTO_DIR)/elastic_training.proto
	$(PROTOC) --proto_path=. --python_out=./ $(DES_PROTO_DIR)/elastic_training.proto
	$(CXX) $(TENSORFLOW_CFLAGS) -std=c++11 -fPIC -I./ -I$(PROTOBUF_INCLUDE) -I$(GRPC_INCLUDE) -c \
		$(DES_PROTO_DIR)/elastic_training.pb.cc -o $(DES_PROTO_DIR)/elastic_training.pb.o
	$(CXX) $(TENSORFLOW_CFLAGS) -std=c++11 -fPIC -I./ -I$(PROTOBUF_INCLUDE) -I$(GRPC_INCLUDE) -c \
		$(DES_PROTO_DIR)/elastic_training.grpc.pb.cc -o $(DES_PROTO_DIR)/elastic_training.grpc.pb.o

DES_CC_SOURCES := $(shell \
        find $(DES_SRC_DIR) -type f \
        \( -name "*.cc" ! -name "*.cu*" \) \
        -exec realpath {} --relative-to . \;)

DES_OBJS := $(DES_CC_SOURCES:.cc=.o)

CU_SOURCES = $(shell \
		find $(DES_SRC_DIR) -type f \
		\( -name "*.cu.cc" \) \
		-exec realpath {} --relative-to . \;)

CU_OBJS := $(CU_SOURCES:.cc=.o)

$(DES_OBJS): %.o:%.cc protobuf grpc des_proto
	mkdir -p $(dir $@)
	$(CXX) $(DES_CFLAGS) $(TENSORFLOW_CFLAGS) $(CXX_CFLAGS) \
        -MP -MMD -MF $<.d -o $@ -c $< -fpic

$(CU_OBJS): %.o:%.cc
	mkdir -p $(dir $@)
	@$(NVCC) -M $< $(DES_CFLAGS) $(TENSORFLOW_CFLAGS) -x cu \
	$(NVCC) \
		-o $@ -c $< $(DES_CFLAGS) $(TENSORFLOW_CFLAGS) -x cu \
		--expt-relaxed-constexpr -Xcompiler -fPIC

ifneq ($(CUDA_CFLAGS),)
$(DES_LIB): $(DES_OBJS) $(CU_OBJS)
	mkdir -p $(dir $@)
	$(CXX) $(DES_CFLAGS) -std=c++11 \
        -o $@ $(DES_PROTO_OBJ) $(DES_OBJS) $(CU_OBJS) \
        $(TENSORFLOW_LDFLAGS) $(DES_LDFLAGS) $(CUDA_LDFLAGS)
else
$(DES_LIB): $(DES_OBJS)
	mkdir -p $(dir $@)
	$(CXX) $(DES_CFLAGS) -std=c++11 -Wl,--version-script=$(DES)/des.script \
        -o $@ $(DES_PROTO_OBJ) $(DES_OBJS) \
        $(TENSORFLOW_LDFLAGS) $(DES_LDFLAGS)
endif
        
