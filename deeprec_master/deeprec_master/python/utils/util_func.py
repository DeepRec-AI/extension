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
""" utils """

from datetime import datetime
import os
import time
import sys
from functools import wraps
import traceback
import pytz
from dateutil.tz import tzlocal

from deeprec_master.python.utils import constants
from deeprec_master.python.utils.logger import logger
from deeprec_master.python.utils import Status

def get_underscore_uppercase_name(camel_case_name):
    lst = []
    for index, char in enumerate(camel_case_name):
        if char.isupper() and index != 0:
            lst.append("_")
        lst.append(char)

    return "".join(lst).upper()


# no explict job completed flag in kubeflow's pytorch crd.
# check all transition, found succeed one means job succeed.
def is_job_completed(job):
    """return if job is completed"""
    job_status = job.get("status", None)
    if job_status is None:
        return False
    conditions = job_status.get("conditions", None)
    if conditions is None:
        return False
    for transition in conditions:
        if transition.get("type", None) == "Succeeded":
            return True
    return False


def get_job_name():
    return os.getenv("JOB_NAME")


def get_tenant_name():
    return os.getenv("NAMESPACE")


def get_utc_nowtime():
    return get_datetime_str(datetime.utcnow())


def get_earliest_time():
    return get_datetime_str(datetime.min)


def get_datetime_str(dt):
    dt.strftime("%Y-%m-%d %H:%M:%S")
    dt_str = str(dt)
    res = dt_str[:10] + "T" + dt_str[11:19] + "Z"
    return res


def get_utc_timestamp(utc_time):
    start_datetime = datetime(1970, 1, 1, tzinfo=tzlocal())
    return (utc_time - start_datetime).total_seconds()


def func_run_time_printer(threshold=None):
    """A decorator that print run time of the function."""

    def wrapper(func):
        @wraps(func)
        def inner_wrapper(*args, **kwargs):
            local_time = time.time()
            result = func(*args, **kwargs)
            duration = time.time() - local_time
            if threshold is None or duration >= threshold:
                logger.info("Function %s run time is %.2fs" % (func.__name__, duration))
            return result

        return inner_wrapper

    return wrapper


def utc2local(utc_dt, tz="Asia/Shanghai"):
    """Get local timestamp from utc."""
    local_tz = pytz.timezone(tz)
    local_dt = utc_dt.replace(tzinfo=pytz.utc).astimezone(local_tz)
    return int(time.mktime(local_dt.timetuple()))


def exit_if_error(st, error_code=-1):
    """Exit if status is error."""
    if st is None:
        logger.warning(
            "Invalid argument, backtrace: {}".format(traceback.print_stack())
        )
    elif isinstance(st, bool):
        if not st:
            sys.exit(error_code)
    elif isinstance(st, Status):
        if not st.ok():
            logger.warning("AIMaster exit with message: {}".format(st.message()))
            sys.exit(error_code)


def get_localtime_str(tz=None):
    """
    Get the local time and return it as a string.

    Args:
      timezone (str, optional): The timezone to use. Defaults to None.
    Returns:
      str: The formatted local time in the "%Y-%m-%dT%H:%M:%S.%f%z" format,
            e.g. "2023-08-30T16:27:20.226956+08:00".
    """
    tz = tz or os.environ.get("TZ", "Asia/Shanghai")
    if tz == "UTC":
        tz = "Asia/Shanghai"

    current_time = datetime.now()
    localized_time = current_time.astimezone(pytz.timezone(tz))

    date_format = "%Y-%m-%dT%H:%M:%S.%f%z"
    formatted_time = localized_time.strftime(date_format)

    # Reformat the timezone offset from "+0800" to "+08:00"
    formatted_time = formatted_time[:-2] + ":" + formatted_time[-2:]
    return formatted_time
