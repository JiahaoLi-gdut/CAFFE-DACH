import os
import glob
import cv2
import caffe
import lmdb
import numpy as np
from caffe.proto import caffe_pb2
from itertools import combinations

source_1 = 'amazon'
source_2 = 'dslr'

def getPlan() :
  # source
  nsamples = 1000
  tranplan = {0:{0:nsamples * 2}, source_1:{}, source_2:{}}
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
  target_1 = 'webcam'
  nsamples = 1000
  testplan = {0:{0:nsamples}, target_1:{}}
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
  return tranplan, testplan

def convertData2Lmdb(datapath) :
  datadict = {}
  domnlist = [x for x in os.listdir(datapath)
              if os.path.isdir(os.path.join(datapath, x))]
  domnlist.sort()
  for domnitem in domnlist:
    labllist = [x for x in os.listdir(os.path.join(datapath, domnitem))
                if os.path.isdir(os.path.join(datapath, domnitem, x))]
    labllist.sort()
    for c, lablitem in enumerate(labllist):
      lablpath = os.path.join(datapath, domnitem, lablitem)
      filelist = glob.iglob(os.path.join(lablpath, '*.jpg'))
      filenumb = 0
      for f in filelist:
        dictkeys = '{}_{}_{}'.format(filenumb, domnitem, c);
        img = cv2.imread(f)
        img = cv2.resize(img, (256, 256))
        img = img.transpose((2, 0, 1))
        datadict[dictkeys] = img
        filenumb = filenumb + 1
  # infodict = getInfoFromDict(datadict)
  # plantupl = getPlanFromDict(infodict)
  tranplan, testplan = getPlan()
  map_size = tranplan[0][0] * 196608 * 8
  env = lmdb.open('image_train_lmdb', map_size = map_size)
  txn = env.begin(write = True)
  count = 0
  for dictkeys in datadict :
    keywords = dictkeys.split('_')
    if keywords[1] not in tranplan : continue
    if int(keywords[2]) not in tranplan[keywords[1]] : continue
    if tranplan[keywords[1]][int(keywords[2])] == 0 : continue
    tranplan[keywords[1]][int(keywords[2])] -= 1
    data = datadict[dictkeys]
    label = int(keywords[2])
    ## data
    datum = caffe_pb2.Datum()
    datum.channels = 3
    datum.height = data.shape[1]
    datum.width = data.shape[2]
    datum.data = data.tostring()
    datum.label = label
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

  map_size = testplan[0][0] * 196608 * 8
  env = lmdb.open('image_test_lmdb', map_size = map_size)
  txn = env.begin(write = True)
  count = 0
  for dictkeys in datadict :
    keywords = dictkeys.split('_')
    if keywords[1] not in testplan : continue
    if int(keywords[2]) not in testplan[keywords[1]] : continue
    if testplan[keywords[1]][int(keywords[2])] == 0 : continue
    testplan[keywords[1]][int(keywords[2])] -= 1
    data = datadict[dictkeys]
    label = int(keywords[2])
    ## data
    datum = caffe_pb2.Datum()
    datum.channels = 3
    datum.height = data.shape[1]
    datum.width = data.shape[2]
    datum.data = data.tostring()
    datum.label = label
    str_id = '{:08}'.format(count)
    txn.put(str_id, datum.SerializeToString())
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
  map_size = tranplan[0][0] * 128 * 8
  env = lmdb.open('image_train_domain_lmdb', map_size = map_size)
  txn = env.begin(write = True)
  count = 0
  for dictkeys in datadict :
    keywords = dictkeys.split('_')
    if keywords[1] not in tranplan : continue
    if int(keywords[2]) not in tranplan[keywords[1]] : continue
    if tranplan[keywords[1]][int(keywords[2])] == 0 : continue
    if keywords[1] not in dom2numb:
      numb = len(dom2numb)
      dom2numb[keywords[1]] = numb
    tranplan[keywords[1]][int(keywords[2])] -= 1
    data = [[[]]]
    label = -1
    if dom2numb[keywords[1]] == source_1: label = 0
    if dom2numb[keywords[1]] == source_2: label = 1
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
  convertData2Lmdb('domain_adaptation_images_resampled')