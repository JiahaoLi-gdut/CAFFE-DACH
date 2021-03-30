import numpy as np
import matplotlib.pyplot as plt
import lmdb
import os
import caffe
import sys
import cv2
from sklearn.manifold import TSNE

CAFFE_ROOT      = '/home/ljh/caffe-master/'
DEPLOY_PROTOTXT = '/home/ljh/caffe-master/examples/_gene_homo/sources_8000_target_4000_dach_logscale_grid/protos/deploy.prototxt'
MODEL_FILE      = '/home/ljh/caffe-master/examples/_gene_homo/sources_8000_target_4000_dach_logscale_grid/snapshots/solver_iter_1000.caffemodel'

# feature visualize
def feat_visualizer(feat_list, mark_list, mark_cmap, mark_hmap):
  mark_clst, mark_hlst = [], []
  for mark in mark_list:
    domain, label = mark
    color = mark_cmap[domain][label]
    hatch = mark_hmap[domain][label]
    mark_clst.append(color)
    mark_hlst.append(hatch)

  point_list = TSNE(learning_rate = 100).fit_transform(feat_list)
  # plt.figure(figsize = (10, 5))
  plt.figure()
  plt.subplot(111)
  #plt.scatter(point_list[:, 0], point_list[:, 1], c = mark_clst, marker = '.')
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
    net.blobs['data'].reshape(1, 1, 1, 13876)
    for key, value in lmdb_cursor:
      datum.ParseFromString(value)
      label = datum.label
      net.blobs['data'].data[...] = caffe.io.datum_to_array(datum)
      out = net.forward(False)
      grid_feat = net.blobs['grid1'].data[0, 0, 0, [4412, 4413, 6518, 6519, 9050, 9051]]
      feat_list.append(grid_feat)
      mark_list.append((lmdb_indx, label))
  return feat_list, mark_list

if __name__ == "__main__":
  lmdb_path_list = []
  lmdb_path_list.append('/home/ljh/caffe-master/examples/_gene_homo/_datasets/gene_10558_lmdb')
  lmdb_path_list.append('/home/ljh/caffe-master/examples/_gene_homo/_datasets/gene_4133_lmdb')
  lmdb_path_list.append('/home/ljh/caffe-master/examples/_gene_homo/_datasets/gene_570_lmdb')
  lmdb_path_list.append('/home/ljh/caffe-master/examples/_gene_homo/_datasets/gene_6102_lmdb')
  lmdb_path_list.append('/home/ljh/caffe-master/examples/_gene_homo/_datasets/gene_6480_lmdb')
  lmdb_path_list.append('/home/ljh/caffe-master/examples/_gene_homo/_datasets/gene_6884_lmdb')
  lmdb_path_list.append('/home/ljh/caffe-master/examples/_gene_homo/_datasets/gene_6947_lmdb')
  
  mark_cmap = {0:{}, 1:{}, 2:{}, 3:{}, 4:{}, 5:{}, 6:{}}
  mark_hmap = {0:{}, 1:{}, 2:{}, 3:{}, 4:{}, 5:{}, 6:{}}
  mark_cmap[0][0], mark_hmap[0][0] = 'r', '.'
  mark_cmap[0][1], mark_hmap[0][1] = 'g', '.'
  mark_cmap[1][0], mark_hmap[1][0] = 'm', '.'
  mark_cmap[1][1], mark_hmap[1][1] = 'b', '.'
  mark_cmap[2][0], mark_hmap[2][0] = 'm', '.'
  mark_cmap[2][1], mark_hmap[2][1] = 'b', '.'
  mark_cmap[3][0], mark_hmap[3][0] = 'm', '.'
  mark_cmap[3][1], mark_hmap[3][1] = 'b', '.'
  mark_cmap[4][0], mark_hmap[4][0] = 'm', '.'
  mark_cmap[4][1], mark_hmap[4][1] = 'b', '.'
  mark_cmap[5][0], mark_hmap[5][0] = 'm', '.'
  mark_cmap[5][1], mark_hmap[5][1] = 'b', '.'
  mark_cmap[6][0], mark_hmap[6][0] = 'm', '.'
  mark_cmap[6][1], mark_hmap[6][1] = 'b', '.'

  net = netInitilizer()
  feat_list, mark_list = netDetailsGetter(net, lmdb_path_list)
  feat_visualizer(feat_list, mark_list, mark_cmap, mark_hmap)