set -o errexit

export DEEPRECMASTER_BUILD_DOCKER_DIR="docker_build/"
export DEEPRECMASTER_IMAGE_TAG=${DEEPRECMASTER_IMAGE_TAG:-`date +%s`}
export DEEPRECMASTER_IMAGE_PATH=${DEEPRECMASTER_IMAGE_PATH:-"registry.cn-shanghai.aliyuncs.com/deeprec-extension/aimaster"}

pushd deeprec_master

mkdir -p ${DEEPRECMASTER_BUILD_DOCKER_DIR}

DEEPRECMASTER_VERSION=$(tools/get_version.sh)
echo DEEPRECMasterVersion: ${DEEPRECMASTER_VERSION}
 
cp dist/deeprec_master-${DEEPRECMASTER_VERSION}-cp36-cp36m-linux_x86_64.whl  ${DEEPRECMASTER_BUILD_DOCKER_DIR}/
cp tools/Dockerfile  ${DEEPRECMASTER_BUILD_DOCKER_DIR}/Dockerfile
cp conf/pip.conf ${DEEPRECMASTER_BUILD_DOCKER_DIR}/
cp deeprecmaster_main.py ${DEEPRECMASTER_BUILD_DOCKER_DIR}/
cp launch_deeprecmaster.sh ${DEEPRECMASTER_BUILD_DOCKER_DIR}/

docker_version=$(sudo docker version | grep "API Version" | awk '{print $3}' | head -n 1)
major_version="$(cut -d'.' -f1 <<<"$docker_version")"
minor_version="$(cut -d'.' -f2 <<<"$docker_version")"
NET_FLAG="net"
if (( $major_version > 1 )) || (( $minor_version > 24 ))
then
  NET_FLAG="network"
fi
pushd ${DEEPRECMASTER_BUILD_DOCKER_DIR}
  IMAGE_FULL_PATH=${DEEPRECMASTER_IMAGE_PATH}:py3.${DEEPRECMASTER_IMAGE_TAG}
  sudo docker build --${NET_FLAG}=host --build-arg DEEPRECMASTER_VERSION=${DEEPRECMASTER_VERSION} -t ${IMAGE_FULL_PATH} .
  echo ${IMAGE_FULL_PATH}
  echo DEEPRECMasterVersion: ${DEEPRECMASTER_VERSION}
  echo Environment: k8s
popd
echo ${IMAGE_FULL_PATH} > ${DEEPRECMASTER_BUILD_DOCKER_DIR}/docker_image_name.txt

popd
