from __future__ import print_function
import caffe
import lmdb
import numpy as np
import scipy.io as sio
from caffe.proto import caffe_pb2
from itertools import combinations
from itertools import permutations

def seeInfoFromDict(infodict) :
  print('{:>8}'.format('(d\\l)'), end = '')
  for lablkeys in infodict[0] :
    print('{:>8}'.format(lablkeys), end = '')
  print('\n', end = '')
  for domnkeys in infodict :
    print('{:>8}'.format(domnkeys), end = '')
    domnvals = infodict[domnkeys]
    for lablkeys in infodict[0] :
      if lablkeys in domnvals :
        print('{:>8}'.format(domnvals[lablkeys]), end = '')
      else :
        print('{:>8}'.format(0), end = '')
    print('\n', end = '')

def seeCombFromList(comblist) :
  combindx = 0
  for combvals in comblist :
    print('#{} train:'.format(combindx), end = '')
    for domnvals in combvals[0] :
      print('{}('.format(domnvals[0]), end = '')
      initflag = True;
      for lablkeys in domnvals[1] :
        if initflag == True :
          initflag = False
          print('{}:{}'.format(lablkeys, domnvals[1][lablkeys]), end = '')
        else :
          print(' {}:{}'.format(lablkeys, domnvals[1][lablkeys]), end = '')
      print(') ', end = '')
    print('; test:', end = '')
    for domnvals in combvals[1] :
      print('{}('.format(domnvals[0]), end = '')
      initflag = True;
      for lablkeys in domnvals[1] :
        if initflag == True :
          initflag = False
          print('{}:{}'.format(lablkeys, domnvals[1][lablkeys]), end = '')
        else :
          print(' {}:{}'.format(lablkeys, domnvals[1][lablkeys]), end = '')
      print(') ', end = '')
    print('\n', end = '')
    combindx += 1

def seePlanFromTupl(plantupl) :
  tranplan, testplan = plantupl
  print('{:>8}'.format('(d\\l)'), end = '')
  for lablkeys in tranplan[0] :
    print('{:>8}'.format(lablkeys), end = '')
  print('\n', end = '')
  for domnkeys in tranplan :
    print('{:>8}'.format(domnkeys), end = '')
    domnvals = tranplan[domnkeys]
    for lablkeys in tranplan[0] :
      if lablkeys in domnvals :
        print('{:>8}'.format(domnvals[lablkeys]), end = '')
      else :
        print('{:>8}'.format(0), end = '')
    print('\n', end = '')
  print('\n{:>8}'.format('(d\\l)'), end = '')
  for lablkeys in testplan[0] :
    print('{:>8}'.format(lablkeys), end = '')
  print('\n', end = '')
  for domnkeys in testplan :
    print('{:>8}'.format(domnkeys), end = '')
    domnvals = testplan[domnkeys]
    for lablkeys in testplan[0] :
      if lablkeys in domnvals :
        print('{:>8}'.format(domnvals[lablkeys]), end = '')
      else :
        print('{:>8}'.format(0), end = '')
    print('\n', end = '')

def seeScalFromTupl(scaltupl) :
  scalmini, scalmaxi = scaltupl
  for scalinfo in scalmini :
    print('{} '.format(scalinfo), end = '')
  print('\n', end = '')
  for scalinfo in scalmaxi :
    print('{} '.format(scalinfo), end = '')
  print('\n', end = '')

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

# def getCombFromRatio(ratio_list) :
#   ratiocom, indexcom = [], []
#   for ratioidx in range(len(ratio_list)) :
#     ratiopmt = list(permutations(ratio_list[ratioidx], len(ratio_list[ratioidx])))
#     new_ratiopmt = list(set(ratiopmt))
#     new_ratiopmt.sort(ratiopmt.index)
#     ratiocom.append(new_ratiopmt)
#     indexcom.append(0, len(new_ratiopmt))
#   ratioidx = 0
#   ratioarr = []
#   while ratioidx < len(indexcom) :
#     if indexcom[ratioidx][0] < indexcom[ratioidx][1] :
#       tempcomb = []
#       for tempindx in range(len(indexcom)) :
#         tempcomb.append(ratiocom[ratioidx][indexcom[tempindx][0]])
#       ratioarr.append(tempcomb)
#       indexcom[ratioidx][0] += 1
#       for backindx in range(ratioidx) :
#         indexcom[backindx][0] = 0
#       ratioidx = -1
#     ratioidx += 1
#   return ratioarr

def getCombFromInfo(infodict, train_domain_size, test_domain_size, train_datum_size, test_datum_size, train_domain_ratio, test_domain_ratio) :
  if train_datum_size  + test_datum_size  > infodict[0][0] : return [], -1
  if train_domain_size + test_domain_size > infodict[1][0] : return [], -2
  domnlist, comblist, combinfo = [], [], []
  for domnkeys in infodict :
    if isinstance(domnkeys, str) :
      domnlist.append(domnkeys)
  trancomb = list(permutations(domnlist, train_domain_size))
  for combindx in range(len(trancomb)-1, -1, -1) :
    samcount = 0
    domnindx = 0
    for domnkeys in trancomb[combindx] :
      sampunit = 0
      lablindx = 0
      for lablkeys in infodict[domnkeys] :
        if isinstance(lablkeys, str) :
          if train_domain_ratio[domnindx][lablindx] > 0 :
            tempunit = infodict[domnkeys][lablkeys] / train_domain_ratio[domnindx][lablindx]
            if sampunit == 0 or tempunit < sampunit :
              sampunit = tempunit
          lablindx += 1
      while lablindx > 0 :
        samcount += sampunit * train_domain_ratio[domnindx][lablindx - 1]
        lablindx -= 1
      domnindx += 1
    if samcount < train_datum_size :
      trancomb.pop(combindx)
      continue
  for combvals in trancomb :
    testdomn = []
    for domnkeys in domnlist :
      if domnkeys not in combvals :
        testdomn.append(domnkeys)
    testcomb = list(combinations(testdomn, test_domain_size))
    for combindx in range(len(testcomb)-1, -1, -1) :
      samcount = 0
      domnindx = 0
      for domnkeys in testcomb[combindx] :
        sampunit = 0
        lablindx = 0
        for lablkeys in infodict[domnkeys] :
          if isinstance(lablkeys, str) :
            if train_domain_ratio[domnindx][lablindx] > 0 :
              tempunit = infodict[domnkeys][lablkeys] / train_domain_ratio[domnindx][lablindx]
              if sampunit == 0 or tempunit < sampunit :
                sampunit = tempunit
            lablindx += 1
        while lablindx > 0 :
          samcount += sampunit * test_domain_ratio[domnindx][lablindx - 1]
          lablindx -= 1
        domnindx += 1
      if samcount >= test_datum_size :
        comblist.append((combvals, testcomb[combindx]))
  for combvals in comblist :
    tranpart, testpart = [], []
    domnindx = 0
    for domnkeys in combvals[0] :
      sampunit = 0
      lablindx = 0
      lablsamp = {}
      for lablkeys in infodict[domnkeys] :
        if isinstance(lablkeys, str) :
          if train_domain_ratio[domnindx][lablindx] != 0 :
            tempunit = infodict[domnkeys][lablkeys] / train_domain_ratio[domnindx][lablindx]
            if sampunit == 0 or tempunit < sampunit :
              sampunit = tempunit
          lablindx += 1
      lablindx = 0
      samcount = 0
      for lablkeys in infodict[domnkeys] :
        if isinstance(lablkeys, str) :
          lablsamp[lablkeys] = sampunit * train_domain_ratio[domnindx][lablindx]
          samcount += lablsamp[lablkeys]
          lablindx += 1
      tranpart.append((domnkeys, lablsamp))
      domnindx += 1
    domnindx = 0
    for domnkeys in combvals[1] :
      sampunit = 0
      lablindx = 0
      lablsamp = {}
      for lablkeys in infodict[domnkeys] :
        if isinstance(lablkeys, str) :
          if test_domain_ratio[domnindx][lablindx] != 0 :
            tempunit = infodict[domnkeys][lablkeys] / test_domain_ratio[domnindx][lablindx]
            if sampunit == 0 or tempunit < sampunit :
              sampunit = tempunit
          lablindx += 1
      lablindx = 0
      samcount = 0
      for lablkeys in infodict[domnkeys] :
        if isinstance(lablkeys, str) :
          lablsamp[lablkeys] = sampunit * test_domain_ratio[domnindx][lablindx]
          samcount += lablsamp[lablkeys]
          lablindx += 1
      testpart.append((domnkeys, lablsamp))
      domnindx += 1
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
    train_datum_size  = int(input('please input the min size of train datum: '))
    test_datum_size   = int(input('please input the min size of test datum: '))
    train_domain_ratio = []
    for domain_index in range(train_domain_size) :
      while True :
        domain_ratio = raw_input('please input the ratio of train domain {} (splited by :)'.format(domain_index))
        ratio_strs = domain_ratio.split(':')
        if len(ratio_strs) == infodict[0][1] :
          break
        else :
          print('the ratio should be splited into {} parts'.format(infodict[0][1]))
      ratio_ints = []
      for ratio_str in ratio_strs :
        ratio_ints.append(int(ratio_str))
      train_domain_ratio.append(ratio_ints)
    test_domain_ratio = []
    for domain_index in range(test_domain_size) :
      while True :
        domain_ratio = raw_input('please input the ratio of test domain {} (splited by :)'.format(domain_index))
        ratio_strs = domain_ratio.split(':')
        if len(ratio_strs) == infodict[0][1] :
          break
        else :
          print('the ratio should be splited into {} parts'.format(infodict[0][1]))
      ratio_ints = []
      for ratio_str in ratio_strs :
        ratio_ints.append(int(ratio_str))
      test_domain_ratio.append(ratio_ints)
    comblist, errorint = getCombFromInfo(infodict, train_domain_size, test_domain_size, train_datum_size, test_datum_size, train_domain_ratio, test_domain_ratio)
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
  datadict = sio.loadmat(datapath)
  datadict.pop('__version__')
  datadict.pop('__header__')
  datadict.pop('__globals__')
  infodict = getInfoFromDict(datadict)
  plantupl = getPlanFromDict(infodict)
  #scaltupl = getScalFromPlan(datadict, plantupl)
  #seeScalFromTupl(scaltupl)
  #datamini = scaltupl[0][4]
  #datamaxi = scaltupl[1][4]
  #usermini = int(input('please input the minimal value for scaling: '))
  #usermaxi = int(input('please input the maximal value for scaling: '))

  tranplan, testplan = cpyPlanFromPlan(plantupl)
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
    txn.put(str_id,datum.SerializeToString())
    count += 1
    if count % 1000 == 0 or count == testplan[0][0] :
      print('test: already handled with {} samples'.format(count))
      txn.commit()
      txn = env.begin(write = True)
    if count == testplan[0][0] : break
  txn.commit()
  env.close()

if __name__=='__main__':
  convertData2Lmdb('SortedGsmPfSexSortedDimDoubleDataSetWithoutMissing.mat')
