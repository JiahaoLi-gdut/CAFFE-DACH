from __future__ import print_function
import caffe
import lmdb
import numpy as np
import scipy.io as sio
from caffe.proto import caffe_pb2
from itertools import combinations
from itertools import permutations

def getPlan() :
  tranplan = {0:{}, 'GPL570':{}}
  tranplan[0][0] = 2000
  tranplan['GPL570']['M'] = 1000
  tranplan['GPL570']['F'] = 1000

  testplan = {0:{}, 'GPL570':{}}
  testplan[0][0] = 4000
  testplan['GPL570']['M'] = 2000
  testplan['GPL570']['F'] = 2000
  return tranplan, testplan

def convertData2Lmdb(datapath) :
  datadict = sio.loadmat(datapath)
  datadict.pop('__version__')
  datadict.pop('__header__')
  datadict.pop('__globals__')

  tranplan, testplan = getPlan()
  map_size = tranplan[0][0] * 20659 * 8
  env = lmdb.open('gene_train_lmdb', map_size = map_size)
  txn = env.begin(write = True)
  count = 0
  for dictkeys in datadict :
    keywords = dictkeys.split('_')
    if keywords[1] not in tranplan : continue
    if keywords[2] not in tranplan[keywords[1]] : continue
    if tranplan[keywords[1]][keywords[2]] == 0 : continue
    tranplan[keywords[1]][keywords[2]] -= 1
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
    if count % 1000 == 0 or count == tranplan[0][0] :
      print('train: already handled with {} samples'.format(count))
      txn.commit()
      txn = env.begin(write = True)
    if count == tranplan[0][0] : break
  txn.commit()
  env.close()

  tranplan, testplan = getPlan()
  map_size = testplan[0][0] * 20659 * 8
  env = lmdb.open('gene_test_lmdb', map_size = map_size)
  txn = env.begin(write = True)
  count = 0
  for dictkeys in datadict :
    keywords = dictkeys.split('_')
    if keywords[1] in tranplan :
      if keywords[2] in tranplan[keywords[1]] :
        if tranplan[keywords[1]][keywords[2]] > 0 :
          tranplan[keywords[1]][keywords[2]] -= 1
          continue
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
    txn.put(str_id,datum.SerializeToString())
    count += 1
    if count % 1000 == 0 or count == testplan[0][0] :
      print('test: already handled with {} samples'.format(count))
      txn.commit()
      txn = env.begin(write = True)
    if count == testplan[0][0] : break
  txn.commit()
  env.close()

  dom2numb = {}
  tranplan, testplan = getPlan()
  map_size = tranplan[0][0] * 13876 * 8
  env = lmdb.open('gene_train_domain_lmdb', map_size = map_size)
  txn = env.begin(write = True)
  count = 0
  for dictkeys in datadict :
    keywords = dictkeys.split('_')
    if keywords[1] not in tranplan : continue
    if keywords[2] not in tranplan[keywords[1]] : continue
    if tranplan[keywords[1]][keywords[2]] == 0 : continue
    if keywords[1] not in dom2numb:
      numb = len(dom2numb)
      dom2numb[keywords[1]] = numb
    tranplan[keywords[1]][keywords[2]] -= 1
    data = [[[]]]
    label = dom2numb[keywords[1]]
    #for dictvals in datadict[dictkeys] :
      #for dataelem in dictvals :
        #dataelem = (dataelem - datamini) / (datamaxi - datamini) * (usermaxi - usermini) + usermini
        #data[0][0].append(float(dataelem))
    data[0][0].append(float(dom2numb[keywords[1]]))
    dataarr = np.array(data)
    datum = caffe_pb2.Datum()
    datum = caffe.io.array_to_datum(dataarr, label)
    str_id = '{:08}'.format(count)
    txn.put(str_id, datum.SerializeToString())
    count += 1
    if count % 1000 == 0 or count == tranplan[0][0] :
      print('train: already handled with {} samples'.format(count))
      txn.commit()
      txn = env.begin(write = True)
    if count == tranplan[0][0] : break
  txn.commit()
  env.close()

if __name__=='__main__':
  convertData2Lmdb('SortedGsmPfSexSortedDimDoubleDataSetWithoutMissing.mat')
