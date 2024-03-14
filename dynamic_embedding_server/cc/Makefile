DES_SRC_DIR := $(DES)/cc/
DES_PROTO_DIR := $(DES)/proto
DES_PROTO_OBJ := $(DES_PROTO_DIR)/elastic_training.pb.o $(DES_PROTO_DIR)/elastic_training.grpc.pb.o 

des_proto: grpc
	mkdir -p $(dir $@)
	@echo "generating pb file"
	$(PROTOC) --cpp_out=./ $(DES_PROTO_DIR)/elastic_training.proto
	$(PROTOC) --grpc_out=./ --plugin=protoc-gen-grpc=$(PROTOC_GRPC_PLUGIN) $(DES_PROTO_DIR)/elastic_training.proto
	$(PROTOC) --proto_path=. --python_out=./ $(DES_PROTO_DIR)/elastic_training.proto
	$(CXX) -std=c++11 -fPIC -I./ -I$(PROTOBUF_INCLUDE) -I$(GRPC_INCLUDE) -c \
		$(DES_PROTO_DIR)/elastic_training.pb.cc -o $(DES_PROTO_DIR)/elastic_training.pb.o
	$(CXX) -std=c++11 -fPIC -I./ -I$(PROTOBUF_INCLUDE) -I$(GRPC_INCLUDE) -c \
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
	$(CXX) $(CFLAGS) $(TENSORFLOW_CFLAGS) $(CXX_CFLAGS) \
        -MP -MMD -MF $<.d -o $@ -c $< -fpic

$(CU_OBJS): %.o:%.cc
	mkdir -p $(dir $@)
	@$(NVCC) -M $< $(CFLAGS) $(TENSORFLOW_CFLAGS) -x cu \
	$(NVCC) \
		-o $@ -c $< $(CFLAGS) $(TENSORFLOW_CFLAGS) -x cu \
		--expt-relaxed-constexpr -Xcompiler -fPIC

ifneq ($(CUDA_CFLAGS),)
$(DES_LIB): $(DES_OBJS) $(CU_OBJS)
	mkdir -p $(dir $@)
	$(CXX) $(CFLAGS) -std=c++11 \
        -o $@ $(DES_PROTO_OBJ) $(DES_OBJS) $(CU_OBJS) \
        $(TENSORFLOW_LDFLAGS) $(LDFLAGS) $(CUDA_LDFLAGS)
else
$(DES_LIB): $(DES_OBJS)
	mkdir -p $(dir $@)
	$(CXX) $(CFLAGS) -std=c++11 -Wl,--version-script=$(DES)/des.script \
        -o $@ $(DES_PROTO_OBJ) $(DES_OBJS) \
        $(TENSORFLOW_LDFLAGS) $(LDFLAGS)
endif
        
