#!/usr/bin/env sh

ROOT=/home/ljh/caffe-master
CAFFE=$ROOT/build/tools/caffe
PREFIX=$ROOT/examples/_gene_homo/sources_4000_target_2000_grl_logscale_conv
SOLVER=$PREFIX/protos/solver.prototxt
LOGDAT=$PREFIX/log_data
GPUID=2

$CAFFE train --solver=$SOLVER --gpu $GPUID 2>&1 | tee $LOGDAT

POSTPY=$PREFIX/scripts/feature_extractor.py
DEPLOY=$PREFIX/protos/deploy.prototxt
MODELS=$PREFIX/snapshots/train_iter_1000.caffemodel
python $POSTPY $ROOT $DEPLOY $MODELS $GPUID