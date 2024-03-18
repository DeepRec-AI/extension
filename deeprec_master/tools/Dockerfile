FROM alideeprec/extension-dev:aimaster-base-py36-ubuntu18.04
ENV TZ="Asia/Shanghai"
ARG DEEPRECMASTER_VERSION

RUN mkdir -p /workspace && mkdir -p /root/.pip && mkdir -p /var/ && mkdir -p /home/pai-aimaster

RUN apt-get update && apt-get install libssl-dev -y libfarmhash-dev libc-ares-dev
COPY pip.conf /root/.pip

RUN pip install numpy grpcio==1.17.0 grpcio-tools==1.17.0 kubernetes==11.0.0 \
    aliyun-python-sdk-core aliyun-python-sdk-ecs jsonpatch

# Note: To reduce job update conflict, replace merge-patch with json-patch,
# but json-patch is not implemented in patch_namespaced_custom_object method for python kubernetes client.
# Here is a workaround, Ref issue: https://github.com/kubernetes-client/python/pull/2040
RUN find / -name "custom_objects_api.py" | xargs sed -i "s/\['application\/merge-patch+json']/['application\/json-patch+json' if isinstance(body, list) else 'application\/merge-patch+json']/g"

COPY deeprec_master-${DEEPRECMASTER_VERSION}-cp36-cp36m-linux_x86_64.whl /workspace
RUN pip install /workspace/deeprec_master-${DEEPRECMASTER_VERSION}-cp36-cp36m-linux_x86_64.whl && rm -rf /workspace/deeprec_master-${DEEPRECMASTER_VERSION}-cp36-cp36m-linux_x86_64.whl

COPY ./deeprecmaster_main.py /var
COPY ./launch_deeprecmaster.sh /var

ENTRYPOINT ["/bin/bash", "/var/launch_deeprecmaster.sh"]
