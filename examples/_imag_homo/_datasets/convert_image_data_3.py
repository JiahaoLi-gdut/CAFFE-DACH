import os
import glob
import cv2
import caffe
import lmdb
import numpy as np
from caffe.proto import caffe_pb2
from itertools import combinations

source_1 = 'dslr'
source_2 = 'webcam'
target_1 = 'amazon'

def getPlan() :
  # source
  nsamples = 2000
  tranplan = {source_1:{}, source_2:{}}
  ratiosum = 0.
  ratiodic = {}
  ratiorng = [9000., 1000.] #[max, min]
  ratiodif = (ratiorng[1] - ratiorng[0]) / 30
  for c in range(0, 31):
    ratiodic[c] = ratiorng[0] + c * ratiodif
    ratiosum += ratiodic[c]
  for c in range(0, 31):
    ratiodic[c] *= nsamples
    ratiodic[c] /= ratiosum
  for c in range(0, 31):
    tranplan[source_1][c] = int(ratiodic[c])
    tranplan[source_2][c] = int(ratiodic[31 - c - 1])
    nsamples -= int(ratiodic[c])
  for c in range(0, 31):
    if nsamples > 0:
      tranplan[source_1][c] += 1
      tranplan[source_2][31 - c - 1] += 1
      nsamples -= 1

  # suma = 0
  # for c in range(0, 31):
  #   suma += tranplan[source_1][c]
  # sumb = 0
  # for c in range(0, 31):
  #   sumb += tranplan[source_2][c]
  # print('{}'.format(suma))
  # print('{}'.format(sumb))

  # target
  nsamples = 2000
  testplan = {target_1:{}}
  while nsamples > 0:
    for c in range(0, 31):
      if nsamples == 0: break
      if c not in testplan[target_1]:
        testplan[target_1][c] = +1
      else:
        testplan[target_1][c] += 1
      nsamples -= 1
  # print(tranplan)
  # print(testplan)
  # waitKey(0)
  return {source_1:tranplan[source_1]}, {source_1:tranplan[source_2]}, testplan

def convertData2Lmdb(datapath) :
  datadict = {}
  domnlist = [x for x in os.listdir(datapath)
              if os.path.isdir(os.path.join(datapath, x))]
  domnlist.sort()
  for domnitem in domnlist:
    datadict[domnitem] = {}
    labllist = [x for x in os.listdir(os.path.join(datapath, domnitem))
                if os.path.isdir(os.path.join(datapath, domnitem, x))]
    labllist.sort()
    for c, lablitem in enumerate(labllist):
      datadict[domnitem][c] = []
      lablpath = os.path.join(datapath, domnitem, lablitem)
      filelist = glob.iglob(os.path.join(lablpath, '*.jpg'))
      filenumb = 0
      for f in filelist:
        img = cv2.imread(f)
        img = cv2.resize(img, (256, 256))
        img = img.transpose((2, 0, 1))
        datadict[domnitem][c].append(img)

  tranplan1, tranplan2, testplan = getPlan()
  map_size = 2000 * 196608 * 8
  env = lmdb.open(source_1 + "_train_0_lmdb", map_size = map_size)
  txn = env.begin(write = True)
  count = 0
  counter = {}
  checker = True
  while checker:
    checker = False
    for domnitem in tranplan1:
      if domnitem not in counter:
        counter[domnitem] = {}
      for c in tranplan1[domnitem]:
        if c not in counter[domnitem]:
          counter[domnitem][c] = 0
        if counter[domnitem][c] < tranplan1[domnitem][c]:
          index = counter[domnitem][c] % len(datadict[domnitem][c]);
          data  = datadict[domnitem][c][index]
          datum = caffe_pb2.Datum()
          datum.channels = 3
          datum.height   = data.shape[1]
          datum.width    = data.shape[2]
          datum.data     = data.tostring()
          datum.label    = c
          str_id = '{:08}'.format(count)
          txn.put(str_id, datum.SerializeToString())
          count += 1
          if count % 1000 == 0:
            print('train: already handled with {} samples'.format(count))
            txn.commit()
            txn = env.begin(write = True)
          counter[domnitem][c] += 1
          checker = True
  txn.commit()
  env.close()

  map_size = 2000 * 196608 * 8
  env = lmdb.open(source_2 + "_train_0_lmdb", map_size = map_size)
  txn = env.begin(write = True)
  count = 0
  counter = {}
  checker = True
  while checker:
    checker = False
    for domnitem in tranplan2:
      if domnitem not in counter:
        counter[domnitem] = {}
      for c in tranplan2[domnitem]:
        if c not in counter[domnitem]:
          counter[domnitem][c] = 0
        if counter[domnitem][c] < tranplan2[domnitem][c]:
          index = counter[domnitem][c] % len(datadict[domnitem][c]);
          data  = datadict[domnitem][c][index]
          datum = caffe_pb2.Datum()
          datum.channels = 3
          datum.height   = data.shape[1]
          datum.width    = data.shape[2]
          datum.data     = data.tostring()
          datum.label    = c
          str_id = '{:08}'.format(count)
          txn.put(str_id, datum.SerializeToString())
          count += 1
          if count % 1000 == 0:
            print('train: already handled with {} samples'.format(count))
            txn.commit()
            txn = env.begin(write = True)
          counter[domnitem][c] += 1
          checker = True
  txn.commit()
  env.close()

  map_size = 2000 * 196608 * 8
  env = lmdb.open(target_1 + "_train_0_lmdb", map_size = map_size)
  txn = env.begin(write = True)
  count = 0
  counter = {}
  checker = True
  while checker:
    checker = False
    for domnitem in testplan:
      if domnitem not in counter:
        counter[domnitem] = {}
      for c in testplan[domnitem]:
        if c not in counter[domnitem]:
          counter[domnitem][c] = 0
        if counter[domnitem][c] < testplan[domnitem][c]:
          index = counter[domnitem][c] % len(datadict[domnitem][c]);
          data  = datadict[domnitem][c][index]
          datum = caffe_pb2.Datum()
          datum.channels = 3
          datum.height   = data.shape[1]
          datum.width    = data.shape[2]
          datum.data     = data.tostring()
          datum.label    = c
          str_id = '{:08}'.format(count)
          txn.put(str_id, datum.SerializeToString())
          count += 1
          if count % 1000 == 0:
            print('train: already handled with {} samples'.format(count))
            txn.commit()
            txn = env.begin(write = True)
          counter[domnitem][c] += 1
          checker = True
  txn.commit()
  env.close()

if __name__=='__main__':
  convertData2Lmdb('domain_adaptation_images')