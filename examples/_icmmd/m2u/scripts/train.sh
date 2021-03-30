#!/usr/bin/env sh

TOOLS=/home/ljh/caffe-master/build/tools
LOG=/home/ljh/caffe-master/examples/_icmmd/m2u/log-data
$TOOLS/caffe train \
    --solver=/home/ljh/caffe-master/examples/_icmmd/m2u/protos/solver.prototxt \
    --gpu 2 2>&1 | tee $LOG
