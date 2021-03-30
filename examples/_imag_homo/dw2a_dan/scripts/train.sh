#!/usr/bin/env sh

TOOLS=/home/ljh/caffe-master/build/tools
LOG=/home/ljh/caffe-master/examples/_imag_homo/dw2a_dan/log-data
$TOOLS/caffe train \
    --solver=/home/ljh/caffe-master/examples/_imag_homo/dw2a_dan/protos/solver.prototxt \
    --weights=/home/ljh/caffe-master/models/bvlc_alexnet/bvlc_alexnet.caffemodel \
    --gpu 2 2>&1 | tee $LOG