#!/usr/bin/env sh

TOOLS=/home/ljh/caffe-master/build/tools
LOG=/home/ljh/caffe-master/examples/_mpd_homo/s2m/log-data
$TOOLS/caffe train \
    --solver=/home/ljh/caffe-master/examples/_mpd_homo/s2m/protos/solver.prototxt \
    --gpu 2 2>&1 | tee $LOG
