#!/usr/bin/env sh

TOOLS=/home/ljh/caffe-master/build/tools
LOG=/home/ljh/caffe-master/examples/_mpd_homo/m2u/log-data
$TOOLS/caffe train \
    --solver=/home/ljh/caffe-master/examples/_mpd_homo/m2u/protos/solver.prototxt \
    --gpu 0 2>&1 | tee $LOG
