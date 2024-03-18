#!/bin/bash
set -o errexit

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 SOURCE_DIRECTORY TARGET_DIRECTORY"
    exit 1
fi

# 读取源目录和目标目录
SOURCE_DIR=$1
TARGET_DIR=$2
echo "$SOURCE_DIR"
# 检查源目录是否存在
if [ ! -d "$SOURCE_DIR" ]; then
    echo "Error: Source directory '$SOURCE_DIR' does not exist."
    exit 1
fi

# 检查目标目录是否存在，不存在则创建
if [ ! -d "$TARGET_DIR" ]; then
    mkdir -p "$TARGET_DIR"
fi

# 使用find搜索并拷贝符合条件的文件
find "$SOURCE_DIR" -type f -name 'elastic_training_pb2*' -exec cp {} "$TARGET_DIR" \;
sed -i 's/tensorflow.core.protobuf/deeprec_master.python.scaling_controller/g' $TARGET_DIR/elastic_training_pb2_grpc.py
echo "Files have been copied to '$TARGET_DIR'."
