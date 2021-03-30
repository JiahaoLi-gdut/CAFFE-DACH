import numpy as np
import matplotlib.pyplot as plt
import lmdb
import os
import caffe
import sys
import cv2

CAFFE_ROOT      = '/home/ljh/caffe-master/'
DEPLOY_PROTOTXT = '/home/ljh/caffe-master/examples/_gene_homo/sources_8000_target_4000_dach_logscale_grid/protos/deploy.prototxt'
MODEL_FILE      = '/home/ljh/caffe-master/examples/_gene_homo/sources_8000_target_4000_dach_logscale_grid/snapshots/solver_iter_1000.caffemodel'

# network initialize
def netInitilizer():
  print 'initilize ...'
  sys.path.insert(0, CAFFE_ROOT + 'python')
  caffe.set_mode_cpu()
  # caffe.set_device(4)
  net = caffe.Net(DEPLOY_PROTOTXT, MODEL_FILE, caffe.TEST)
  return net

# get params from network and data from net.blobs
def netDetailsGetter(lmdb_path, net, grid_feat_dict):
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
    grid_feat = net.blobs['grid1'].data[0, 0, 0, 6518]
    lpfc_feat = 1 if net.params['lp_fc'][0].data[0, 6518] > 0 else -1

    if label not in grid_feat_dict :
      grid_feat_dict[label] = []
      grid_feat_dict[label].append(grid_feat * lpfc_feat)
    else :
      grid_feat_dict[label].append(grid_feat * lpfc_feat)
  lmdb_env.close()

  return grid_feat_dict

# def grid_feat_dict_to_boxplot(grid_feat_dict):
#   key_list, value_list = [], []
#   for key in grid_feat_dict:
#     key_list.append(key)
#     value_list.append(grid_feat_dict[key])
#   plt.boxplot(value_list, labels = key_list);
#   plt.show()

# def grid_feat_dict_to_hist(grid_feat_dict):
  # plt.subplot(2, 1, 1)
  # plt.xlim(-1, 1)
  # plt.hist(grid_feat_dict[0], bins = 1000)
  # plt.subplot(2, 1, 2)
  # plt.xlim(-1, 1)
  # plt.hist(grid_feat_dict[1], bins = 1000)
  # plt.subplot(2, 1, 1)
  # plt.xlim(-0.4, 0.4)
  # plt.hist(grid_feat_dict[0], bins=128, alpha=0.75, edgecolor='None', facecolor='red', histtype='bar', hatch='/')
  # plt.hist(grid_feat_dict[1], bins=128, alpha=0.75, edgecolor='None', facecolor='green', histtype='bar', hatch='+')
  # plt.show()

# def visualize_grid_feat_of_lmdb(lmdb_path, net):
#   grid_feat_dict = {}
#   netDetailsGetter(lmdb_path, net, grid_feat_dict)
#   grid_feat_dict_to_boxplot(grid_feat_dict)
#   grid_feat_dict_to_hist(grid_feat_dict)

def visualize_grid_feat_of_list(lmdb_path_list, net):
  count = 1
  listsize = len(lmdb_path_list)
  for lmdb_path in lmdb_path_list :
    plt.figure(figsize=(9, 1))
    grid_feat_dict = {}
    domain = lmdb_path.split("_")[4]
    netDetailsGetter(lmdb_path, net, grid_feat_dict)
    ax = plt.subplot(1, 1, 1)
    ax.tick_params(axis='x', labelsize=13, labelcolor='k')
    # ax.set_xlabel("Domian: GPL" + domain, fontsize=13)
    ax.tick_params(axis='y', labelsize=13)
    ax.grid(color='k', linestyle='--', linewidth=1, alpha=0.1)
    ax.set_xlim(-1.0, 1.0)
    if count < listsize:
      ax.hist(grid_feat_dict[0], bins=32, alpha=0.75, facecolor='m', hatch='x')
      ax.hist(grid_feat_dict[1], bins=32, alpha=0.75, facecolor='b', hatch='+')
    else:
      ax.hist(grid_feat_dict[0], bins=32, alpha=0.75, facecolor='r', hatch='x')
      ax.hist(grid_feat_dict[1], bins=32, alpha=0.75, facecolor='g', hatch='+')
    count += 1
    plt.subplots_adjust(left = 0.06, bottom = 0.24, right = 0.98, top = 0.96, wspace = 0, hspace = 0)
    plt.savefig('hist_GPL' + domain + '.pdf', format='pdf', dpi=300, pad_inches = 0)
    # plt.show()
  #grid_feat_dict_to_boxplot(grid_feat_dict)
  # grid_feat_dict_to_hist(grid_feat_dict)

if __name__ == "__main__":
  net = netInitilizer()
  lmdb_path_list = []
  lmdb_path_list.append('/home/ljh/caffe-master/examples/_gene_homo/_datasets/gene_4133_lmdb')
  lmdb_path_list.append('/home/ljh/caffe-master/examples/_gene_homo/_datasets/gene_570_lmdb')
  lmdb_path_list.append('/home/ljh/caffe-master/examples/_gene_homo/_datasets/gene_6102_lmdb')
  lmdb_path_list.append('/home/ljh/caffe-master/examples/_gene_homo/_datasets/gene_6480_lmdb')
  lmdb_path_list.append('/home/ljh/caffe-master/examples/_gene_homo/_datasets/gene_6884_lmdb')
  lmdb_path_list.append('/home/ljh/caffe-master/examples/_gene_homo/_datasets/gene_6947_lmdb')
  lmdb_path_list.append('/home/ljh/caffe-master/examples/_gene_homo/_datasets/gene_10558_lmdb')
  #for lmdb_path in lmdb_path_list:
  #  visualize_grid_feat_of_lmdb(lmdb_path, net)
  visualize_grid_feat_of_list(lmdb_path_list, net)
