#!/usr/bin/env bash
set -ex

PYTHON=python
export CUDA_VISIBLE_DEVICES=

TEST_FILES=( \
  ./python/redistribution_ops_test.py
)

echo "Test file include "${TEST_FILES[@]};
for test_file in ${TEST_FILES[@]}
do
  echo "#########################################################################"
  echo "Run test file "${test_file};
  ${PYTHON} -u ${test_file}
done

