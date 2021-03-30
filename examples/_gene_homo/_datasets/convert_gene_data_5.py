from __future__ import print_function
import caffe
import lmdb
import numpy as np
import scipy.io as sio
from caffe.proto import caffe_pb2
from itertools import combinations
from itertools import permutations

def getPlan() :
  tranplan1 = {0:{}, 'GPL6947':{}}
  tranplan1[0][0] = 2000
  tranplan1['GPL6947']['M'] = 1000
  tranplan1['GPL6947']['F'] = 1000

  tranplan2 = {0:{}, 'GPL10558':{}}
  tranplan2[0][0] = 2000
  tranplan2['GPL10558']['M'] = 1000
  tranplan2['GPL10558']['F'] = 1000

  testplan = {0:{}, 'GPL570':{}}
  testplan[0][0] = 2000
  testplan['GPL570']['M'] = 1000
  testplan['GPL570']['F'] = 1000
  return tranplan1, tranplan2, testplan

def convertData2Lmdb(datapath) :
  datadict = sio.loadmat(datapath)
  datadict.pop('__version__')
  datadict.pop('__header__')
  datadict.pop('__globals__')

  tranplan1, tranplan2, testplan = getPlan()
  map_size = tranplan1[0][0] * 20659 * 8
  env = lmdb.open('gene_train1_lmdb', map_size = map_size)
  txn = env.begin(write = True)
  count = 0
  for dictkeys in datadict :
    keywords = dictkeys.split('_')
    if keywords[1] not in tranplan1 : continue
    if keywords[2] not in tranplan1[keywords[1]] : continue
    if tranplan1[keywords[1]][keywords[2]] == 0 : continue
    tranplan1[keywords[1]][keywords[2]] -= 1
    data = [[[]]]
    label = 0 if keywords[2] == 'F' else 1
    for dictvals in datadict[dictkeys] :
      for dataelem in dictvals :
        #dataelem = (dataelem - datamini) / (datamaxi - datamini) * (usermaxi - usermini) + usermini
        data[0][0].append(float(dataelem))
    dataarr = np.array(data)
    datum = caffe_pb2.Datum()
    datum = caffe.io.array_to_datum(dataarr, label)
    str_id = '{:08}'.format(count)
    txn.put(str_id, datum.SerializeToString())
    count += 1
    if count % 1000 == 0 or count == tranplan1[0][0] :
      print('train: already handled with {} samples'.format(count))
      txn.commit()
      txn = env.begin(write = True)
    if count == tranplan1[0][0] : break
  txn.commit()
  env.close()

  map_size = tranplan2[0][0] * 20659 * 8
  env = lmdb.open('gene_train2_lmdb', map_size = map_size)
  txn = env.begin(write = True)
  count = 0
  for dictkeys in datadict :
    keywords = dictkeys.split('_')
    if keywords[1] not in tranplan2 : continue
    if keywords[2] not in tranplan2[keywords[1]] : continue
    if tranplan2[keywords[1]][keywords[2]] == 0 : continue
    tranplan2[keywords[1]][keywords[2]] -= 1
    data = [[[]]]
    label = 0 if keywords[2] == 'F' else 1
    for dictvals in datadict[dictkeys] :
      for dataelem in dictvals :
        #dataelem = (dataelem - datamini) / (datamaxi - datamini) * (usermaxi - usermini) + usermini
        data[0][0].append(float(dataelem))
    dataarr = np.array(data)
    datum = caffe_pb2.Datum()
    datum = caffe.io.array_to_datum(dataarr, label)
    str_id = '{:08}'.format(count)
    txn.put(str_id, datum.SerializeToString())
    count += 1
    if count % 1000 == 0 or count == tranplan2[0][0] :
      print('train: already handled with {} samples'.format(count))
      txn.commit()
      txn = env.begin(write = True)
    if count == tranplan2[0][0] : break
  txn.commit()
  env.close()

  map_size = testplan[0][0] * 20659 * 8
  env = lmdb.open('gene_test_lmdb', map_size = map_size)
  txn = env.begin(write = True)
  count = 0
  for dictkeys in datadict :
    keywords = dictkeys.split('_')
    if keywords[1] not in testplan : continue
    if keywords[2] not in testplan[keywords[1]] : continue
    if testplan[keywords[1]][keywords[2]] == 0 : continue
    testplan[keywords[1]][keywords[2]] -= 1
    data = [[[]]]
    label = 0 if keywords[2] == 'F' else 1
    for dictvals in datadict[dictkeys] :
      for dataelem in dictvals :
        #dataelem = (dataelem - datamini) / (datamaxi - datamini) * (usermaxi - usermini) + usermini
        data[0][0].append(float(dataelem))
    dataarr = np.array(data)
    datum = caffe_pb2.Datum()
    datum = caffe.io.array_to_datum(dataarr, label)
    str_id = '{:08}'.format(count)
    txn.put(str_id, datum.SerializeToString())
    count += 1
    if count % 1000 == 0 or count == testplan[0][0] :
      print('train: already handled with {} samples'.format(count))
      txn.commit()
      txn = env.begin(write = True)
    if count == testplan[0][0] : break
  txn.commit()
  env.close()

if __name__=='__main__':
  convertData2Lmdb('SortedGsmPfSexSortedDimDoubleDataSetWithoutMissing.mat')
