# Copyright 2024 The DeepRec Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
""" k8s config """

import os

import kubernetes.client
from kubernetes.config import kube_config
from deeprec_master.python.utils.logger import logger


def get_k8s_apiclient():
    """return a k8s api client"""
    master_config = kubernetes.client.Configuration()
    use_service_account = True
    if (
        os.getenv("IN_CLUSTER_CONFIG") is not None
        and os.environ["IN_CLUSTER_CONFIG"] == "true"
    ):
        use_service_account = False
    if use_service_account:
        logger.info("using master service account.")
        api_server_url = os.getenv("TENANT_API_SERVER_URL")
        if api_server_url is None:
            api_server_url = (
                "https://"
                + os.environ["KUBERNETES_SERVICE_HOST"]
                + ":"
                + os.environ["KUBERNETES_SERVICE_PORT"]
            )
        with open(
            "/var/run/secrets/kubernetes.io/serviceaccount/token", "r", encoding="utf-8"
        ) as f:
            token = f.read()
        # Specify the endpoint of your Kube cluster
        master_config.host = api_server_url
        # Security part.
        # In this simple example we are not going to verify the SSL certificate of
        # the remote cluster (for simplicity reason)
        master_config.verify_ssl = False
        # Nevertheless if you want to do it you can with these 2 parameters
        # configuration.verify_ssl=True
        # ssl_ca_cert is the filepath to the file that contains the certificate.
        # configuration.ssl_ca_cert="certificate"
        master_config.api_key = {"authorization": "Bearer " + token}
    else:
        kube_config.load_kube_config(client_configuration=master_config)
        master_config.assert_hostname = False
    # Create a ApiClient with our confi
    return kubernetes.client.ApiClient(master_config)


# dist-mnist-nativeam-worker-0
def get_pod_name(jobname, tasktype, taskindex):
    return jobname + "-" + tasktype + "-" + str(taskindex)
