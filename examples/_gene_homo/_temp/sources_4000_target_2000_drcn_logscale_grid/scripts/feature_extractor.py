import numpy as np
import os
import caffe
import sys
import heapq

CAFFE_ROOT      = sys.argv[1]
DEPLOY_PROTOTXT = sys.argv[2]
MODEL_FILE      = sys.argv[3]
GPUID           = sys.argv[4]

def netInitilizer():
  print 'initilize ...'
  sys.path.insert(0, CAFFE_ROOT + 'python')
  caffe.set_device(int(GPUID))
  caffe.set_mode_gpu()
  net = caffe.Net(DEPLOY_PROTOTXT, MODEL_FILE, caffe.TEST)
  return net

def netDetailsGetter(net):
  feat0 = net.params['lp_fc'][0].data[0]
  feat1 = net.params['lp_fc'][0].data[1]
  feat2 = abs(feat0)
  nmaxi = heapq.nlargest(5, xrange(len(feat2)), feat2.take)
  nmaxd = heapq.nlargest(5, feat2)
  print nmaxi
  print nmaxd

if __name__ == "__main__":
  net = netInitilizer()
  netDetailsGetter(net)