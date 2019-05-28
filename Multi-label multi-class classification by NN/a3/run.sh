#!/bin/sh
pip3 install --user numpy
pip3 install --user scipy
pip3 install --user sklearn
pip3 install --user pickle-mixin
python3 predict.py $1 $2