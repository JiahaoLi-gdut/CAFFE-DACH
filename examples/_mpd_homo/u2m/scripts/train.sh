#!/usr/bin/env sh

TOOLS=/home/ljh/caffe-master/build/tools
LOG=/home/ljh/caffe-master/examples/_mpd_homo/u2m/log-data
$TOOLS/caffe train \
    --solver=/home/ljh/caffe-master/examples/_mpd_homo/u2m/protos/solver.prototxt \
    --gpu 3 2>&1 | tee $LOG
