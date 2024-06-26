MASTER_SRC_DIR := $(MASTER)/$(MASTER)/core

MASTER_PROTO_DIR := $(MASTER)/$(MASTER)/proto
MASTER_PROTO_OBJ := $(MASTER_PROTO_DIR)/error_code.pb.o \
		    $(MASTER_PROTO_DIR)/data_distributor/data_config.pb.o \
		    $(MASTER_PROTO_DIR)/data_distributor/data_manager.pb.o \
		    $(MASTER_PROTO_DIR)/model_ready/model_ready.pb.o \
		    $(MASTER_PROTO_DIR)/elastic_training.pb.o \
		    $(MASTER_PROTO_DIR)/data_distributor/data_manager.grpc.pb.o \
		    $(MASTER_PROTO_DIR)/elastic_training.grpc.pb.o \
		    $(MASTER_PROTO_DIR)/model_ready/model_ready.grpc.pb.o

proto: grpc
	mkdir -p $(dir $@)
	@echo "generating pb file"
	$(PROTOC) --cpp_out=./$(MASTER)/ -I$(MASTER) $(MASTER_PROTO_DIR)/error_code.proto
	$(PROTOC) --cpp_out=./$(MASTER)/ -I$(MASTER) $(MASTER_PROTO_DIR)/data_distributor/data_config.proto
	$(PROTOC) --cpp_out=./$(MASTER)/ -I$(MASTER) $(MASTER_PROTO_DIR)/data_distributor/data_manager.proto
	$(PROTOC) --cpp_out=./$(MASTER)/ -I$(MASTER) $(MASTER_PROTO_DIR)/elastic_training.proto
	$(PROTOC) --cpp_out=./$(MASTER)/ -I$(MASTER) $(MASTER_PROTO_DIR)/model_ready/model_ready.proto

	$(PROTOC) --grpc_out=./$(MASTER)/ -I$(MASTER) --plugin=protoc-gen-grpc=$(PROTOC_GRPC_CPP_PLUGIN) $(MASTER_PROTO_DIR)/data_distributor/data_manager.proto
	$(PROTOC) --grpc_out=./$(MASTER)/ -I$(MASTER) --plugin=protoc-gen-grpc=$(PROTOC_GRPC_PYTHON_PLUGIN) $(MASTER_PROTO_DIR)/elastic_training.proto
	$(PROTOC) --grpc_out=./$(MASTER)/ -I$(MASTER) --plugin=protoc-gen-grpc=$(PROTOC_GRPC_CPP_PLUGIN) $(MASTER_PROTO_DIR)/elastic_training.proto
	$(PROTOC) --grpc_out=./$(MASTER)/ -I$(MASTER) --plugin=protoc-gen-grpc=$(PROTOC_GRPC_CPP_PLUGIN) $(MASTER_PROTO_DIR)/model_ready/model_ready.proto

	$(PROTOC) --proto_path=./$(MASTER)/ -I$(MASTER) --python_out=./$(MASTER) $(MASTER_PROTO_DIR)/error_code.proto
	$(PROTOC) --proto_path=./$(MASTER)/ -I$(MASTER) --python_out=./$(MASTER) $(MASTER_PROTO_DIR)/data_distributor/data_config.proto
	$(PROTOC) --proto_path=./$(MASTER)/ -I$(MASTER) --python_out=./$(MASTER) $(MASTER_PROTO_DIR)/elastic_training.proto

	$(CXX) $(TENSORFLOW_CFLAGS) -std=c++11 -fPIC -I$(MASTER) -I$(PROTOBUF_INCLUDE) -I$(GRPC_INCLUDE) -c $(MASTER_PROTO_DIR)/error_code.pb.cc -o $(MASTER_PROTO_DIR)/error_code.pb.o
	$(CXX) $(TENSORFLOW_CFLAGS) -std=c++11 -fPIC -I$(MASTER) -I$(PROTOBUF_INCLUDE) -I$(GRPC_INCLUDE) -c $(MASTER_PROTO_DIR)/data_distributor/data_config.pb.cc -o $(MASTER_PROTO_DIR)/data_distributor/data_config.pb.o
	$(CXX) $(TENSORFLOW_CFLAGS) -std=c++11 -fPIC -I$(MASTER) -I$(PROTOBUF_INCLUDE) -I$(GRPC_INCLUDE) -c $(MASTER_PROTO_DIR)/data_distributor/data_manager.pb.cc -o $(MASTER_PROTO_DIR)/data_distributor/data_manager.pb.o
	$(CXX) $(TENSORFLOW_CFLAGS) -std=c++11 -fPIC -I$(MASTER) -I$(PROTOBUF_INCLUDE) -I$(GRPC_INCLUDE) -c $(MASTER_PROTO_DIR)/data_distributor/data_manager.grpc.pb.cc -o $(MASTER_PROTO_DIR)/data_distributor/data_manager.grpc.pb.o
	$(CXX) $(TENSORFLOW_CFLAGS) -std=c++11 -fPIC -I$(MASTER) -I$(PROTOBUF_INCLUDE) -I$(GRPC_INCLUDE) -c $(MASTER_PROTO_DIR)/elastic_training.pb.cc -o $(MASTER_PROTO_DIR)/elastic_training.pb.o
	$(CXX) $(TENSORFLOW_CFLAGS) -std=c++11 -fPIC -I$(MASTER) -I$(PROTOBUF_INCLUDE) -I$(GRPC_INCLUDE) -c $(MASTER_PROTO_DIR)/elastic_training.grpc.pb.cc -o $(MASTER_PROTO_DIR)/elastic_training.grpc.pb.o
	$(CXX) $(TENSORFLOW_CFLAGS) -std=c++11 -fPIC -I$(MASTER) -I$(PROTOBUF_INCLUDE) -I$(GRPC_INCLUDE) -c $(MASTER_PROTO_DIR)/model_ready/model_ready.pb.cc -o $(MASTER_PROTO_DIR)/model_ready/model_ready.pb.o
	$(CXX) $(TENSORFLOW_CFLAGS) -std=c++11 -fPIC -I$(MASTER) -I$(PROTOBUF_INCLUDE) -I$(GRPC_INCLUDE) -c $(MASTER_PROTO_DIR)/model_ready/model_ready.grpc.pb.cc -o $(MASTER_PROTO_DIR)/model_ready/model_ready.grpc.pb.o

MASTER_CC_SOURCES := $(shell \
        find $(MASTER_SRC_DIR) -type f \
        \( -name "*.cc" ! -name "*.cu*" \) \
        -exec realpath {} --relative-to . \;)

MASTER_OBJS := $(MASTER_CC_SOURCES:.cc=.o)

MASTER_CFLAGS := $(CFLAGS)
MASTER_CFLAGS += -I$(GRPC_INCLUDE)
MASTER_CFLAGS += -I$(PROTOBUF_INCLUDE)
MASTER_CFLAGS += -I$(ABSEIL_INCLUDE)
MASTER_CFLAGS += -I$(RAPIDJSON_INCLUDE)
MASTER_CFLAGS += -I$(MASTER)

MASTER_LDFLAGS := $(LDFLAGS)
MASTER_LDFLAGS += -L$(PROTOBUF_LIB)
MASTER_LDFLAGS += -L$(GRPC_LIB)
MASTER_LDFLAGS += -L$(ABSEIL_LIB)
MASTER_LDFLAGS += -L$(RAPIDJSON_LIB)
MASTER_LDFLAGS += -lprotobuf
MASTER_LDFLAGS += -lgrpc++
MASTER_LDFLAGS += -lgrpc
MASTER_LDFLAGS += -lssl
MASTER_LDFLAGS += -labsl
MASTER_LDFLAGS += -ldl
MASTER_LDFLAGS += -lpthread
MASTER_LDFLAGS += -lcares
MASTER_LDFLAGS += -lz

$(MASTER_OBJS): %.o:%.cc protobuf grpc proto pybind abseil rapidjson
	mkdir -p $(dir $@)
	$(CXX) $(MASTER_CFLAGS) $(CXX_CFLAGS) $(TENSORFLOW_CFLAGS)\
        -MP -MMD -MF $<.d -o $@ -c $< -fpic

$(MASTER_LIB): $(MASTER_OBJS)
	mkdir -p $(dir $@)
	$(CXX) $(MASTER_CFLAGS) $(TENSORFLOW_CFLAGS) -std=c++11 -o $@ $(MASTER_PROTO_OBJ) $(MASTER_OBJS) $(MASTER_LDFLAGS)
