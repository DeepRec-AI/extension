FROM alideeprec/extension-dev:cpu-py36-ubuntu18.04

RUN mkdir /workspace/
COPY ./*.whl /tmp/
COPY ./train.py /workspace/
RUN pip install /tmp/*.whl
RUN rm -f /tmp/*.whl
