GAZER_SRC_DIR := $(GAZER)/$(GAZER)/cc
GAZER_CC_SOURCES := $(shell \
        find $(GAZER_SRC_DIR)/client $(GAZER_SRC_DIR)/graph $(GAZER_SRC_DIR)/ops $(GAZER_SRC_DIR)/util -type f \
        \( -name "*.cc" ! -name "*.cu*" \) \
        -exec realpath {} --relative-to . \;)

GAZER_PROTO_DIR := $(GAZER_SRC_DIR)/proto
GAZER_PROTO_OBJ := $(GAZER_PROTO_DIR)/scheduler.pb.o $(GAZER_PROTO_DIR)/scheduler.grpc.pb.o

GAZER_CFLAGS := $(CFLAGS)
GAZER_CFLAGS += -I$(PROTOBUF_INCLUDE)
GAZER_CFLAGS += -I$(GRPC_INCLUDE)
GAZER_CFLAGS += -I$(RAPIDJSON_INCLUDE)
GAZER_CFLAGS += -I$(GAZER)

GAZER_LDFLAGS := $(LDFLAGS)
GAZER_LDFLAGS += -L$(PROTOBUF_LIB)
GAZER_LDFLAGS += -L$(GRPC_LIB)
GAZER_LDFLAGS += -lprotobuf
GAZER_LDFLAGS += -lgrpc++
GAZER_LDFLAGS += -lgrpc
GAZER_LDFLAGS += -lssl
GAZER_LDFLAGS += -lcrypto

gazer_proto: grpc
	mkdir -p $(dir $@)
	@echo "generating pb file"
	$(PROTOC) --cpp_out=./ $(GAZER_PROTO_DIR)/scheduler.proto
	$(PROTOC) --grpc_out=./ --plugin=protoc-gen-grpc=$(PROTOC_GRPC_PLUGIN) $(GAZER_PROTO_DIR)/scheduler.proto
	$(CXX) $(TENSORFLOW_CFLAGS) -std=c++11 -fpic -I. -I$(PROTOBUF_INCLUDE) -I$(GRPC_INCLUDE) \
		-c $(GAZER_PROTO_DIR)/scheduler.pb.cc -o $(GAZER_PROTO_DIR)/scheduler.pb.o
	$(CXX) $(TENSORFLOW_CFLAGS) -std=c++11 -fpic -I. -I$(PROTOBUF_INCLUDE) -I$(GRPC_INCLUDE) \
		-c $(GAZER_PROTO_DIR)/scheduler.grpc.pb.cc -o $(GAZER_PROTO_DIR)/scheduler.grpc.pb.o

GAZER_OBJS := $(GAZER_CC_SOURCES:.cc=.o)

$(GAZER_OBJS): %.o:%.cc protobuf grpc gazer_proto rapidjson
	mkdir -p $(dir $@)
	$(CXX) $(GAZER_CFLAGS) $(CXX_CFLAGS) $(TENSORFLOW_CFLAGS)\
        -MP -MMD -MF $<.d -o $@ -c $< -fpic

$(GAZER_LIB): $(GAZER_OBJS)
	mkdir -p $(dir $@)
	$(CXX) $(GAZER_CFLAGS) -std=c++11 -Wl,--version-script=$(GAZER)/gazer.script \
        -o $@ $(GAZER_OBJS) $(GAZER_PROTO_OBJ) \
        $(TENSORFLOW_LDFLAGS) \
	$(GAZER_LDFLAGS) -lcares
