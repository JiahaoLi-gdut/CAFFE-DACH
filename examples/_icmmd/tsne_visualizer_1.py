import numpy as np
import matplotlib.pyplot as plt
import os
import lmdb
import caffe
import sys
import cv2
from sklearn.manifold import TSNE

CAFFE_ROOT      = '/home/ljh/caffe-master/'
DEPLOY_PROTOTXT = '/home/ljh/caffe-master/examples/_icmmd/deploy.prototxt_1'
MODEL_FILE      = '/home/ljh/caffe-master/examples/_icmmd/u2m/snapshots/train_iter_2000.caffemodel'

# feature visualize
def feat_visualizer(feat_list, mark_list, mark_cmap, mark_hmap):
  mark_clst, mark_hlst = [], []
  for mark in mark_list:
    domain = mark
    color = mark_cmap[domain]
    hatch = mark_hmap[domain]
    mark_clst.append(color)
    mark_hlst.append(hatch)

  point_list = TSNE(learning_rate=100, n_iter=1000).fit_transform(feat_list)
  plt.figure()
  plt.subplot(111)
  for i in range(0, len(point_list)):
    plt.scatter(point_list[i, 0], point_list[i, 1], c = mark_clst[i], marker = mark_hlst[i])
  plt.legend(loc=2)
  plt.show()

# network initialize
def netInitilizer():
  print 'initilize ...'
  sys.path.insert(0, CAFFE_ROOT + 'python')
  caffe.set_mode_cpu()
  # caffe.set_device(4)
  net = caffe.Net(DEPLOY_PROTOTXT, MODEL_FILE, caffe.TEST)
  return net

# get params from network and data from net.blobs
def netDetailsGetter(net, lmdb_path_list):
  feat_list, mark_list = [], []
  for lmdb_indx, lmdb_path in enumerate(lmdb_path_list):
    lmdb_env = lmdb.open(lmdb_path)
    lmdb_txn = lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor()
    datum = caffe.proto.caffe_pb2.Datum()
    net.blobs['data'].reshape(1, 1, 28, 28)
    count = 0
    for key, value in lmdb_cursor:
      datum.ParseFromString(value)
      label = datum.label
      net.blobs['data'].data[...] = caffe.io.datum_to_array(datum)
      out = net.forward(False)
      fc3_feat = net.blobs['fc3'].data[0, ...]
      feat_list.append(fc3_feat)
      mark_list.append(lmdb_indx)
      count += 1
      if count == 7438: break
  return feat_list, mark_list

if __name__ == "__main__":
  spath = '/home/ljh/caffe-master/examples/_icmmd/_datasets/mnist_train_lmdb'
  tpath = '/home/ljh/caffe-master/examples/_icmmd/_datasets/usps_test_lmdb'
  # simgs = [os.path.join(spath, cdir, item)
  #          for cdir in os.listdir(spath)
  #          for item in os.listdir(os.path.join(spath, cdir))]
  # timgs = [os.path.join(tpath, cdir, item)
  #          for cdir in os.listdir(tpath)
  #          for item in os.listdir(os.path.join(tpath, cdir))]
  mark_cmap, mark_hmap = {}, {}
  mark_cmap[0], mark_hmap[0] = 'k', '.'
  mark_cmap[1], mark_hmap[1] = 'k', 'x'
  net = netInitilizer()
  feat_list, mark_list = netDetailsGetter(net, [spath, tpath])
  feat_visualizer(feat_list, mark_list, mark_cmap, mark_hmap)