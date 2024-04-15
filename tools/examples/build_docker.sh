#!/bin/bash
set -o errexit

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BUILD_DOCKER_DIR="${SCRIPT_DIR}/docker_build"

mkdir -p ${BUILD_DOCKER_DIR}

cp ${SCRIPT_DIR}/../../gazer/dist/gazer-1.0-cp36-cp36m-linux_x86_64.whl ${BUILD_DOCKER_DIR}
cp ${SCRIPT_DIR}/../../dynamic_embedding_server/dist/dynamic_embedding_server-1.0-cp36-cp36m-linux_x86_64.whl ${BUILD_DOCKER_DIR}
cp ${SCRIPT_DIR}/../../tf_fault_tolerance/dist/tf_fault_tolerance-1.0-cp36-cp36m-linux_x86_64.whl ${BUILD_DOCKER_DIR}
cp ${SCRIPT_DIR}/../../deeprec_master/dist/deeprec_master-1.0-cp36-cp36m-linux_x86_64.whl ${BUILD_DOCKER_DIR}
cp ${SCRIPT_DIR}/../dockerfiles/Dockerfile ${BUILD_DOCKER_DIR}
cp ${SCRIPT_DIR}/train.py ${BUILD_DOCKER_DIR}

pushd ${BUILD_DOCKER_DIR}
  IMAGE_FULL_PATH="registry.cn-shanghai.aliyuncs.com/deeprec-extension/extension:v1.0"
  sudo docker build --network=host -t ${IMAGE_FULL_PATH} .
  echo ${IMAGE_FULL_PATH}
  echo Environment: k8s
popd
echo ${IMAGE_FULL_PATH} > ${BUILD_DOCKER_DIR}/docker_image_name.txt

