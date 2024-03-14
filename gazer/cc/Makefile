GAZER_SRC_DIR := $(GAZER)/cc
GAZER_CC_SOURCES := $(shell \
        find $(GAZER_SRC_DIR)/client $(GAZER_SRC_DIR)/graph $(GAZER_SRC_DIR)/ops $(GAZER_SRC_DIR)/util -type f \
        \( -name "*.cc" ! -name "*.cu*" \) \
        -exec realpath {} --relative-to . \;)

GAZER_PROTO_DIR := $(GAZER_SRC_DIR)/proto
GAZER_PROTO_OBJ := $(GAZER_PROTO_DIR)/scheduler.pb.o $(GAZER_PROTO_DIR)/scheduler.grpc.pb.o

gazer_proto: grpc
	mkdir -p $(dir $@)
	@echo "generating pb file"
	$(PROTOC) --cpp_out=./ $(GAZER_PROTO_DIR)/scheduler.proto
	$(PROTOC) --grpc_out=./ --plugin=protoc-gen-grpc=$(PROTOC_GRPC_PLUGIN) $(GAZER_PROTO_DIR)/scheduler.proto
	$(CXX) $(GLIBCXX_USE_CXX11_ABI) -std=c++11 -fpic -I. -I$(PROTOBUF_INCLUDE) -I$(GRPC_INCLUDE) \
		-c $(GAZER_PROTO_DIR)/scheduler.pb.cc -o $(GAZER_PROTO_DIR)/scheduler.pb.o
	$(CXX) $(GLIBCXX_USE_CXX11_ABI) -std=c++11 -fpic -I. -I$(PROTOBUF_INCLUDE) -I$(GRPC_INCLUDE) \
		-c $(GAZER_PROTO_DIR)/scheduler.grpc.pb.cc -o $(GAZER_PROTO_DIR)/scheduler.grpc.pb.o

GAZER_OBJS := $(GAZER_CC_SOURCES:.cc=.o)

$(GAZER_OBJS): %.o:%.cc protobuf grpc gazer_proto rapidjson
	mkdir -p $(dir $@)
	$(CXX) $(CFLAGS) $(CXX_CFLAGS) $(TENSORFLOW_CFLAGS)\
        -MP -MMD -MF $<.d -o $@ -c $< -fpic

$(GAZER_LIB): $(GAZER_OBJS)
	mkdir -p $(dir $@)
	$(CXX) $(CFLAGS) -std=c++11 -Wl,--version-script=$(GAZER)/gazer.script \
        -o $@ $(GAZER_OBJS) $(GAZER_PROTO_OBJ) \
        $(TENSORFLOW_LDFLAGS) \
		$(LDFLAGS) -lcares

