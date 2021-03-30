#!/usr/bin/env sh

TOOLS=/home/ljh/caffe-master/build/tools
LOG=/home/ljh/caffe-master/examples/_mpd_homo/m2s/log-data
$TOOLS/caffe train \
    --solver=/home/ljh/caffe-master/examples/_mpd_homo/m2s/protos/solver.prototxt \
    --gpu 1 2>&1 | tee $LOG
