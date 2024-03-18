#!/bin/bash

echo [`date +"%F %T,%3N"` DeepRecMaster] INFO: Launch arguments: $@

python /var/deeprecmaster_main.py $@