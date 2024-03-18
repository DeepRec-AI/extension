#!/bin/bash
set -e

pip install pylint==2.12.2

PYTHONPATH=${PWD}:${PYTHONPATH}

# TMP_FILE="py_files.txt"
# find $(realpath .)/aimaster/ -name "*.py" > $TMP_FILE
# while read line; 
# do
#     echo $line;
#     pylint --rcfile=.pylintrc --output-format=parseable $line;
# done < $TMP_FILE
# rm -rf $TMP_FILE

pylint --rcfile=.pylintrc --output-format=parseable --jobs=8 $( find deeprec_master/ -type f -name '*.py')