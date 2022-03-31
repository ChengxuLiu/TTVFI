#!/usr/bin/env bash

echo "Need pytorch>=1.0.0"
# source activate pytorch1.0.0

# export PYTHONPATH=$PYTHONPATH:$(pwd)/../../../my_package
# PATH=$PYTHONPATH:$(pwd)/../../..
# export PYTHONPATH=$PYTHONPATH:.//lib/python3.6/site-packages/

rm -rf build *.egg-info dist
# python setup.py install
# python setup.py install --user
pip install -e .
# python setup.py install --prefix=./
# python setup.py install --prefix=.//lib/python3.6/site-packages/
# export PYTHONPATH=PATH