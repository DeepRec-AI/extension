#!/usr/bin/env bash
set -x

kill $(ps aux | grep '[p]ython e2e/aimaster_up.py' | awk '{print $2}')
kill $(ps aux | grep '[p]ython e2e/basic_test.py' | awk '{print $2}')

TF_CONFIG="{\"cluster\": {\"worker\": [\"localhost:2222\"], \"ps\": [\"localhost:10086\", \"localhost:10087\", \"localhost:10088\"], \"chief\": [\"localhost:2220\"]},\"task\": {\"type\": \"chief\", \"index\": 0}}" python e2e/aimaster_up.py > aimaster.log 2>&1 &

export DEEPRECMASTER_ADDR=localhost:60001
export CUDA_VISIBLE_DEVICES=

## SCALING UP PS 
TF_CONFIG="{\"cluster\": {\"worker\": [\"localhost:2222\"], \"ps\": [\"localhost:10086\", \"localhost:10087\", \"localhost:10088\", \"localhost:10089\", \"localhost:10090\",  \"localhost:10091\"], \"chief\": [\"localhost:2220\"]},\"task\": {\"type\": \"ps\", \"index\": 3}}" python e2e/basic_test.py > ps_3.log 2>&1 &
sleep 1
TF_CONFIG="{\"cluster\": {\"worker\": [\"localhost:2222\"], \"ps\": [\"localhost:10086\", \"localhost:10087\", \"localhost:10088\", \"localhost:10089\",  \"localhost:10090\",  \"localhost:10091\"], \"chief\": [\"localhost:2220\"]},\"task\": {\"type\": \"ps\", \"index\": 4}}" python e2e/basic_test.py > ps_4.log 2>&1 &
sleep 1
TF_CONFIG="{\"cluster\": {\"worker\": [\"localhost:2222\"], \"ps\": [\"localhost:10086\", \"localhost:10087\", \"localhost:10088\", \"localhost:10089\",  \"localhost:10090\",  \"localhost:10091\"], \"chief\": [\"localhost:2220\"]},\"task\": {\"type\": \"ps\", \"index\": 5}}" python e2e/basic_test.py > ps_5.log 2>&1 &

##ACTUAL WORKER AND PS
TF_CONFIG="{\"cluster\": {\"worker\": [\"localhost:2222\"], \"ps\": [\"localhost:10086\", \"localhost:10087\", \"localhost:10088\"], \"chief\": [\"localhost:2220\"]},\"task\": {\"type\": \"ps\", \"index\": 0}}" python e2e/basic_test.py > ps_0.log 2>&1 &
sleep 1
TF_CONFIG="{\"cluster\": {\"worker\": [\"localhost:2222\"], \"ps\": [\"localhost:10086\", \"localhost:10087\", \"localhost:10088\"], \"chief\": [\"localhost:2220\"]},\"task\": {\"type\": \"ps\", \"index\": 1}}" python e2e/basic_test.py > ps_1.log 2>&1 &
sleep 1
TF_CONFIG="{\"cluster\": {\"worker\": [\"localhost:2222\"], \"ps\": [\"localhost:10086\", \"localhost:10087\", \"localhost:10088\"], \"chief\": [\"localhost:2220\"]},\"task\": {\"type\": \"ps\", \"index\": 2}}" python e2e/basic_test.py > ps_2.log 2>&1 &
sleep 1
TF_CONFIG="{\"cluster\": {\"worker\": [\"localhost:2222\"], \"ps\": [\"localhost:10086\", \"localhost:10087\", \"localhost:10088\"], \"chief\": [\"localhost:2220\"]},\"task\": {\"type\": \"worker\", \"index\": 0}}" python e2e/basic_test.py > worker.log 2>&1 &
sleep 1 
TF_DUMP_GRAPH_PREFIX=./ TF_CONFIG="{\"cluster\": {\"worker\": [\"localhost:2222\"], \"ps\": [\"localhost:10086\", \"localhost:10087\", \"localhost:10088\"], \"chief\": [\"localhost:2220\"]},\"task\": {\"type\": \"chief\", \"index\": 0}}" python e2e/basic_test.py

kill $(ps aux | grep '[p]ython e2e/aimaster_up.py' | awk '{print $2}')
kill $(ps aux | grep '[p]ython e2e/basic_test.py' | awk '{print $2}')
