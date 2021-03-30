from __future__ import print_function
import caffe
import lmdb
import numpy as np
import scipy.io as sio
from caffe.proto import caffe_pb2
from itertools import combinations
from itertools import permutations

def getPlan() :
  tranplan1 = {0:{}, 'GPL6102':{}}
  tranplan1[0][0] = 847
  tranplan1['GPL6102']['M'] = 383
  tranplan1['GPL6102']['F'] = 464

  tranplan2 = {0:{}, 'GPL6884':{}}
  tranplan2[0][0] = 1432
  tranplan2['GPL6884']['M'] = 716
  tranplan2['GPL6884']['F'] = 716

  tranplan3 = {0:{}, 'GPL4133':{}}
  tranplan3[0][0] = 1431
  tranplan3['GPL4133']['M'] = 716
  tranplan3['GPL4133']['F'] = 715

  tranplan4 = {0:{}, 'GPL6947':{}}
  tranplan4[0][0] = 1430
  tranplan4['GPL6947']['M'] = 715
  tranplan4['GPL6947']['F'] = 715

  tranplan5 = {0:{}, 'GPL6480':{}}
  tranplan5[0][0] = 1430
  tranplan5['GPL6480']['M'] = 715
  tranplan5['GPL6480']['F'] = 715

  tranplan6 = {0:{}, 'GPL570':{}}
  tranplan6[0][0] = 1430
  tranplan6['GPL570']['M'] = 715
  tranplan6['GPL570']['F'] = 715

  testplan = {0:{}, 'GPL570':{}}
  testplan[0][0] = 4000
  testplan['GPL570']['M'] = 1898
  testplan['GPL570']['F'] = 2102
  return tranplan1, tranplan2, tranplan3, tranplan4, tranplan5, tranplan6, testplan

def convertData2Lmdb(datapath) :
  datadict = sio.loadmat(datapath)
  datadict.pop('__version__')
  datadict.pop('__header__')
  datadict.pop('__globals__')

  allplans = getPlan()
  for index, currplan in enumerate(allplans):
    map_size = currplan[0][0] * 20659 * 8
    env = lmdb.open('gene_'+ str(index) + '_lmdb', map_size = map_size)
    txn = env.begin(write = True)
    count = 0
    for dictkeys in datadict :
      keywords = dictkeys.split('_')
      if keywords[1] not in currplan : continue
      if keywords[2] not in currplan[keywords[1]] : continue
      if currplan[keywords[1]][keywords[2]] == 0 : continue
      currplan[keywords[1]][keywords[2]] -= 1
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
      if count % 1000 == 0 or count == currplan[0][0] :
        print('train: already handled with {} samples'.format(count))
        txn.commit()
        txn = env.begin(write = True)
      if count == currplan[0][0] : break
    txn.commit()
    env.close()

if __name__=='__main__':
  convertData2Lmdb('SortedGsmPfSexSortedDimDoubleDataSetWithoutMissing.mat')
