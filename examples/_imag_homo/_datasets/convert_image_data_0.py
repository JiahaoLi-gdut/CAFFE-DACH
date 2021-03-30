import os
import glob
import cv2
import caffe
import lmdb
import numpy as np
from caffe.proto import caffe_pb2
from itertools import combinations

def seeInfoFromDict(infodict) :
  print '{:>8}'.format('(d\\l)'),
  for lablkeys in infodict[0] :
    print '{:>8}'.format(lablkeys),
  print '\n',
  for domnkeys in infodict :
    print '{:>8}'.format(domnkeys),
    domnvals = infodict[domnkeys]
    for lablkeys in infodict[0] :
      if lablkeys in domnvals :
        print '{:>8}'.format(domnvals[lablkeys]),
      else :
        print '{:>8}'.format(0),
    print '\n',

def seeCombFromList(comblist) :
  combindx = 0
  for combvals in comblist :
    print '#{} train:'.format(combindx),
    for domnvals in combvals[0] :
      print '{}({})'.format(domnvals[0], domnvals[1]),
    print '; test:',
    for domnvals in combvals[1] :
      print '{}({})'.format(domnvals[0], domnvals[1]),
    print '\n',
    combindx += 1

def seePlanFromTupl(plantupl) :
  tranplan, testplan = plantupl
  print '{:>8}'.format('(d\\l)'),
  for lablkeys in tranplan[0] :
    print '{:>8}'.format(lablkeys),
  print '\n',
  for domnkeys in tranplan :
    print '{:>8}'.format(domnkeys),
    domnvals = tranplan[domnkeys]
    for lablkeys in tranplan[0] :
      if lablkeys in domnvals :
        print '{:>8}'.format(domnvals[lablkeys]),
      else :
        print '{:>8}'.format(0),
    print '\n',
  print '\n{:>8}'.format('(d\\l)'),
  for lablkeys in testplan[0] :
    print '{:>8}'.format(lablkeys),
  print '\n',
  for domnkeys in testplan :
    print '{:>8}'.format(domnkeys),
    domnvals = testplan[domnkeys]
    for lablkeys in testplan[0] :
      if lablkeys in domnvals :
        print '{:>8}'.format(domnvals[lablkeys]),
      else :
        print '{:>8}'.format(0),
    print '\n',

def seeScalFromTupl(scaltupl) :
  scalmini, scalmaxi = scaltupl
  for scalinfo in scalmini :
    print '{} '.format(scalinfo),
  print '\n',
  for scalinfo in scalmaxi :
    print '{} '.format(scalinfo),
  print '\n',

def cpyPlanFromPlan(plantupl) :
  sorctran, sorctest = plantupl
  trgttran, trgttest = {}, {}
  for domnkeys in sorctran :
    if domnkeys not in trgttran :
      trgttran[domnkeys] = {}
    for lablkeys in sorctran[domnkeys] :
      trgttran[domnkeys][lablkeys] = sorctran[domnkeys][lablkeys]   
  for domnkeys in sorctest :
    if domnkeys not in trgttest :
      trgttest[domnkeys] = {}
    for lablkeys in sorctest[domnkeys] :
      trgttest[domnkeys][lablkeys] = sorctest[domnkeys][lablkeys]
  return trgttran, trgttest
'''
       label
    a b c d 0 1
d a . . . . . .
o b . . . . . .
m c . . . . . .
a d . . . . . .
i 0 . . . . s l
n 1 . . . . d c
///////////////
s: sample count
d: domain count
l: label  count
c: label-domain count
'''
def getInfoFromDict(datadict) :
  infodict = {}
  infodict[0] = {}
  infodict[1] = {}
  infodict[0][0] = 0
  infodict[0][1] = 0
  infodict[1][0] = 0
  infodict[1][1] = 0
  for dictkeys in datadict :
    keywords = dictkeys.split('_')
    infodict[0][0] += 1
    if keywords[2] not in infodict[0] :
      infodict[0][1] += 1
      infodict[0][keywords[2]] = +1
      infodict[1][keywords[2]] = +0
    else :
      infodict[0][keywords[2]] += 1
    if keywords[1] not in infodict :
      infodict[1][0] += 1
      infodict[1][1] += 1
      infodict[1][keywords[2]] += 1
      infodict[keywords[1]] = {}
      infodict[keywords[1]][0] = +1
      infodict[keywords[1]][1] = +1
      infodict[keywords[1]][keywords[2]] = +1
    elif keywords[2] not in infodict[keywords[1]] :
      infodict[1][1] += 1
      infodict[1][keywords[2]] += 1
      infodict[keywords[1]][0] += 1
      infodict[keywords[1]][1] += 1
      infodict[keywords[1]][keywords[2]] = +1
    else :
      infodict[keywords[1]][0] += 1
      infodict[keywords[1]][keywords[2]] += 1
  return infodict

def getCombFromInfo(infodict, train_domain_size, test_domain_size, train_datum_size, test_datum_size) :
  if train_datum_size  + test_datum_size  > infodict[0][0] : return [], -1
  if train_domain_size + test_domain_size > infodict[1][0] : return [], -2
  domnlist, comblist, combinfo = [], [], []
  for domnkeys in infodict :
    if isinstance(domnkeys, str) :
      domnlist.append(domnkeys)
  trancomb = list(combinations(domnlist, train_domain_size))
  for combindx in range(len(trancomb)-1, -1, -1) :
    samcount = 0
    for domnkeys in trancomb[combindx] :
      samcount += infodict[domnkeys][0]
    if samcount < train_datum_size :
      trancomb.pop(combindx)
  for combvals in trancomb :
    testdomn = []
    for domnkeys in domnlist :
      if domnkeys not in combvals :
        testdomn.append(domnkeys)
    testcomb = list(combinations(testdomn, test_domain_size))
    for combindx in range(len(testcomb)-1, -1, -1) :
      samcount = 0
      for domnkeys in testcomb[combindx] :
        samcount += infodict[domnkeys][0]
      if samcount >= test_datum_size :
        comblist.append((combvals, testcomb[combindx]))
  for combvals in comblist :
    tranpart, testpart = [], []
    for domnkeys in combvals[0] :
      tranpart.append((domnkeys, infodict[domnkeys][0]))
    for domnkeys in combvals[1] :
      testpart.append((domnkeys, infodict[domnkeys][0]))
    combinfo.append((tuple(tranpart), tuple(testpart)))
  return combinfo, 0

def getPlanFromComb(infodict, domncomb, train_datum_size, test_datum_size) :
  trancomb, testcomb = domncomb
  tranlist, tranplan = [], {}
  testlist, testplan = [], {}
  for combvals in trancomb :
    for lablkeys in infodict[combvals[0]] :
      if isinstance(lablkeys, str) :
        samcount = infodict[combvals[0]][lablkeys]
        tranlist.append([combvals[0], lablkeys, samcount, 0])
  while train_datum_size > 0 :
    for planvals in tranlist :
      if train_datum_size > 0 :
        if planvals[2] > 0 :
          planvals[2] -= 1
          planvals[3] += 1
          train_datum_size -= 1
        else :
          continue
      else :
        break
  tranplan[0], tranplan[1] = {}, {}
  tranplan[0][0], tranplan[0][1] = 0, 0
  tranplan[1][0], tranplan[1][1] = 0, 0
  for planvals in tranlist :
    tranplan[0][0] += planvals[3]
    if planvals[1] not in tranplan[0] :
      tranplan[0][1] += 1
      tranplan[0][planvals[1]] = +planvals[3]
      tranplan[1][planvals[1]] = +0
    else :
      tranplan[0][planvals[1]] += planvals[3]
    if planvals[0] not in tranplan :
      tranplan[1][0] += 1
      tranplan[1][1] += 1
      tranplan[1][planvals[1]] += 1
      tranplan[planvals[0]] = {}
      tranplan[planvals[0]][0] = +planvals[3]
      tranplan[planvals[0]][1] = +1
      tranplan[planvals[0]][planvals[1]] = +planvals[3]
    elif planvals[1] not in tranplan[planvals[0]] :
      tranplan[1][1] += 1
      tranplan[1][planvals[1]] += 1
      tranplan[planvals[0]][0] += planvals[3]
      tranplan[planvals[0]][1] += 1
      tranplan[planvals[0]][planvals[1]] = +planvals[3]
    else :
      tranplan[planvals[0]][0] += planvals[3]
      tranplan[planvals[0]][planvals[1]] += planvals[3]

  for combvals in testcomb :
    for lablkeys in infodict[combvals[0]] :
      if isinstance(lablkeys, str) :
        samcount = infodict[combvals[0]][lablkeys]
        testlist.append([combvals[0], lablkeys, samcount, 0])
  while test_datum_size > 0 :
    for planvals in testlist :
      if test_datum_size > 0 :
        if planvals[2] > 0 :
          planvals[2] -= 1
          planvals[3] += 1
          test_datum_size -= 1
        else :
          continue
      else :
        break
  testplan[0], testplan[1] = {}, {}
  testplan[0][0], testplan[0][1] = 0, 0
  testplan[1][0], testplan[1][1] = 0, 0
  for planvals in testlist :
    testplan[0][0] += planvals[3]
    if planvals[1] not in testplan[0] :
      testplan[0][1] += 1
      testplan[0][planvals[1]] = +planvals[3]
      testplan[1][planvals[1]] = +0
    else :
      testplan[0][planvals[1]] += planvals[3]
    if planvals[0] not in testplan :
      testplan[1][0] += 1
      testplan[1][1] += 1
      testplan[1][planvals[1]] += 1
      testplan[planvals[0]] = {}
      testplan[planvals[0]][0] = +planvals[3]
      testplan[planvals[0]][1] = +1
      testplan[planvals[0]][planvals[1]] = +planvals[3]
    elif planvals[1] not in testplan[planvals[0]] :
      testplan[1][1] += 1
      testplan[1][planvals[1]] += 1
      testplan[planvals[0]][0] += planvals[3]
      testplan[planvals[0]][1] += 1
      testplan[planvals[0]][planvals[1]] = +planvals[3]
    else :
      testplan[planvals[0]][0] += planvals[3]
      testplan[planvals[0]][planvals[1]] += planvals[3]
  return tranplan, testplan

def getPlanFromDict(infodict) :
  while True :
    seeInfoFromDict(infodict)
    train_domain_size = int(input('please input the size of train domain: '))
    test_domain_size  = int(input('please input the size of test domain: '))
    train_datum_size  = int(input('please input the size of train datum: '))
    test_datum_size   = int(input('please input the size of test datum: '))
    comblist, errorint = getCombFromInfo(infodict, train_domain_size, test_domain_size, train_datum_size, test_datum_size)
    if errorint == -1 :
      print('error: train_datum_size({})  + test_datum_size({})  > total_datum_size({})'.format(train_datum_size,   test_datum_size,  infodict[0][0]))
      continue
    elif errorint == -2 :
      print('error: train_domain_size({}) + test_domain_size({}) > total_domain_size({})'.format(train_domain_size, test_domain_size, infodict[1][0]))
      continue
    elif not comblist :
      print('error: In these parameters, no combination found')
      continue
    while True :
      seeCombFromList(comblist)
      usrinput = int(input('please choose combination for your plan: '))
      if usrinput >= 0 and usrinput < len(comblist) :
        plantupl = getPlanFromComb(infodict, comblist[usrinput], train_datum_size, test_datum_size)
        seePlanFromTupl(plantupl)
        confirm  = raw_input('press Y/y for confirmation: ')
        if confirm == 'Y' or confirm == 'y' : return plantupl
        continue

def getScalFromPlan(datadict, plantupl) :
  tranplan, testplan = cpyPlanFromPlan(plantupl)
  miniinfo, maxiinfo = [], []
  miniindx, maxiindx = 0, 0
  minidata, maxidata = 0, 0
  datcount = 0
  for dictkeys in datadict :
    keywords = dictkeys.split('_')
    if keywords[1] not in tranplan : continue
    if keywords[2] not in tranplan[keywords[1]] : continue
    if tranplan[keywords[1]][keywords[2]] == 0 : continue
    tranplan[keywords[1]][keywords[2]] -= 1
    currindx = 0
    for dictvals in datadict[dictkeys] :
      for dataelem in dictvals :
        if datcount == 0 :
          miniinfo, maxiinfo = keywords, keywords
          miniindx, maxiindx = 0, 0
          minidata, maxidata = dataelem, dataelem
        else :
          if dataelem < minidata :
            miniinfo = keywords
            miniindx = currindx
            minidata = dataelem
          if dataelem > maxidata :
            maxiinfo = keywords
            maxiindx = currindx
            maxidata = dataelem
        currindx += 1
        datcount += 1

  datcount = 0
  for dictkeys in datadict :
    keywords = dictkeys.split('_')
    if keywords[1] not in testplan : continue
    if keywords[2] not in testplan[keywords[1]] : continue
    if testplan[keywords[1]][keywords[2]] == 0 : continue
    testplan[keywords[1]][keywords[2]] -= 1
    currindx = 0
    for dictvals in datadict[dictkeys] :
      for dataelem in dictvals :
        if datcount == 0 :
          miniinfo, maxiinfo = keywords, keywords
          miniindx, maxiindx = 0, 0
          minidata, maxidata = dataelem, dataelem
        else :
          if dataelem < minidata :
            miniinfo = keywords
            miniindx = currindx
            minidata = dataelem
          if dataelem > maxidata :
            maxiinfo = keywords
            maxiindx = currindx
            maxidata = dataelem
        currindx += 1
        datcount += 1
  miniinfo.append(miniindx)
  miniinfo.append(minidata)
  maxiinfo.append(maxiindx)
  maxiinfo.append(maxidata)
  return tuple(miniinfo), tuple(maxiinfo)

def convertData2Lmdb(datapath) :
  datadict = {}
  domnlist = [x for x in os.listdir(datapath)
              if os.path.isdir(os.path.join(datapath, x))]
  domnlist.sort()
  for _, domnitem in enumerate(domnlist):
    labllist = [x for x in os.listdir(os.path.join(datapath, domnitem))
                if os.path.isdir(os.path.join(datapath, domnitem, x))]
    labllist.sort()
    for c, lablitem in enumerate(labllist):
      lablpath = os.path.join(datapath, domnitem, lablitem)
      filelist = glob.iglob(os.path.join(lablpath, '*.jpg'))
      filenumb = 0
      for f in filelist:
        dictkeys = '{}_{}_c{}'.format(filenumb, domnitem, c);
        img = cv2.imread(f)
        img = cv2.resize(img, (256, 256))
        img = img.transpose((2, 0, 1))
        datadict[dictkeys] = img
        filenumb = filenumb + 1
  infodict = getInfoFromDict(datadict)
  plantupl = getPlanFromDict(infodict)
  tranplan, testplan = cpyPlanFromPlan(plantupl)
  map_size = tranplan[0][0] * 196608 * 8
  env = lmdb.open('image_train_lmdb', map_size = map_size)
  txn = env.begin(write = True)
  count = 0
  for dictkeys in datadict :
    keywords = dictkeys.split('_')
    if keywords[1] not in tranplan : continue
    if keywords[2] not in tranplan[keywords[1]] : continue
    if tranplan[keywords[1]][keywords[2]] == 0 : continue
    tranplan[keywords[1]][keywords[2]] -= 1
    data = datadict[dictkeys]
    label = int(keywords[2].lstrip('c'))
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
    if keywords[2] not in testplan[keywords[1]] : continue
    if testplan[keywords[1]][keywords[2]] == 0 : continue
    testplan[keywords[1]][keywords[2]] -= 1
    data = datadict[dictkeys]
    label = int(keywords[2].lstrip('c'))
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
  tranplan, testplan = cpyPlanFromPlan(plantupl)
  map_size = tranplan[0][0] * 128 * 8
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
    label = -1
    if dom2numb[keywords[1]] == 'amazon': label = 0
    if dom2numb[keywords[1]] == 'dslr':   label = 1
    if dom2numb[keywords[1]] == 'webcam': label = 2
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
    convertData2Lmdb('domain_adaptation_images')
