import logging
import os
import sys

from deeprec_master.python.utils.logger import logger
from deeprec_master.python.utils.args import parse_args
from deeprec_master.python.deeprecmaster import TfJobMaster
def main():
    logger.info("Deeprecmaster is getting started!")

    job_name = os.getenv('JOB_NAME', None)
    namespace = os.getenv('NAMESPACE', None)

    args = parse_args()
    logger.info(args)
    
    if job_name is not None and namespace is not None:
        logger.info("Create TfJobMaster for namespace: %s, job_name: %s",
                    namespace, job_name)
        job_controller = TfJobMaster(job_name, namespace, args)
    else:
        logger.error("JOB_NAME or NAMESPACE or JOB_TYPE not set, exit...")
        sys.exit(-1)
    
    job_controller.start()
    job_controller.submit_plan()
    job_controller.join()


if __name__ == "__main__":
    main()