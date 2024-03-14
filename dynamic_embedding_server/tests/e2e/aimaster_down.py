from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from concurrent import futures
from threading import Lock
import grpc
import logging
import base64
import os
import json
import time
import sys

from dynamic_embedding_server.proto import elastic_training_pb2_grpc
from dynamic_embedding_server.proto import elastic_training_pb2

mutex = Lock()
class ElasticTrainingServicer(elastic_training_pb2_grpc.ElasticTrainingServiceServicer):
    def __init__(self):
        tf_config = os.environ.get("TF_CONFIG", {})
        tf_config_json = json.loads(tf_config)
        ps_array = tf_config_json["cluster"]["ps"]
        worker_array = tf_config_json["cluster"]["worker"]
        self.stub_list = []
        for addr in ps_array:
            self.stub_list.append(elastic_training_pb2_grpc.ElasticTrainingServiceStub(grpc.insecure_channel(addr)))
        for addr in worker_array:
            self.stub_list.append(elastic_training_pb2_grpc.ElasticTrainingServiceStub(grpc.insecure_channel(addr)))
        self.stub_list.append(elastic_training_pb2_grpc.ElasticTrainingServiceStub(grpc.insecure_channel(tf_config_json["cluster"]["chief"][0])))
        
        self._count = 0
        self._update_count = 0
        self._is_scale = False

    def IsReadyScaling(self, request, context):
        if self._is_scale == True:
          return
        logging.info("message is {}".format(request))
        resp = elastic_training_pb2.IsReadyScalingResponse()
        resp.scaling_action = elastic_training_pb2.SCALING_DOWN
        resp.ps_num = 2
        mutex.acquire()
        self._count += 1
        mutex.release()
        while self._count != 2:
          time.sleep(1)
        return resp

    def ReadyToUpdate(self, request, context):
        cluster_json = {"cluster": {"ps": ["localhost:10086", "localhost:10087"]}}
        cluster_str = json.dumps(cluster_json)
        mutex.acquire()
        self._update_count += 1
        mutex.release()
        while self._update_count != 2:
          time.sleep(1)
        mutex.acquire()
        if self._is_scale == False:
          for i in range(0, len(self.stub_list)):
            if i == 2 or i == 3: continue
            stub = self.stub_list[i]
            _req = elastic_training_pb2.UpdateServerDefRequest()
            _req.cluster_def = cluster_str
            stub.UpdateServerDef(_req)
          self._is_scale = True
        mutex.release()
        return elastic_training_pb2.ReadyToUpdateResponse()

def start_grpc_server():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    elastic_training_pb2_grpc.add_ElasticTrainingServiceServicer_to_server(
        ElasticTrainingServicer(), server)
    port = server.add_insecure_port('[::]:60001')
    logging.info('Setup remote communication, listening on port: {}'.format(port))
    server.start()
    server.wait_for_termination()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    start_grpc_server()
