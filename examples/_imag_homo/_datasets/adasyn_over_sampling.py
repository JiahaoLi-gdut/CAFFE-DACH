import os
import glob
import cv2
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import ADASYN

def read_data(datapath):
  datadict, labldict = {}, {}
  domnlist = [x for x in os.listdir(datapath)
              if os.path.isdir(os.path.join(datapath, x))]
  for domnitem in domnlist:
    labldict[domnitem] = {}
    datadict[domnitem] = {'X': [], 'Y': []}
    labllist = [x for x in os.listdir(os.path.join(datapath, domnitem))
                if os.path.isdir(os.path.join(datapath, domnitem, x))]
    for lablcode, lablitem in enumerate(labllist):
      labldict[domnitem][lablcode] = lablitem
      lablpath = os.path.join(datapath, domnitem, lablitem)
      filelist = glob.iglob(os.path.join(lablpath, '*.jpg'))
      print('Loading data with label {} in domain {}'.format(lablitem, domnitem))
      for fileitem in filelist:
        imagdata = cv2.imread(fileitem)
        imagdata = cv2.resize(imagdata, (256, 256))
        imagdata = imagdata.reshape(256 * 256 * 3)
        #imagdata = imagdata.transpose((2, 0, 1))
        datadict[domnitem]['X'].append(imagdata)
        datadict[domnitem]['Y'].append(lablcode)
  return datadict, labldict

def resampling(datadict, labldict, savepath):
  ratiodic = {}
  for domnitem in datadict:
    ratiodic[domnitem] = {}
  for lablcode in range(0, 31):
    ratiodic['amazon'][lablcode] = 145
    ratiodic['dslr'][lablcode]   = 100
    ratiodic['webcam'][lablcode] = 100
  
  for domnitem in datadict:
    lablcout, lablnumb = {}, {}
    sorcdata = datadict[domnitem]['X']
    sorclabl = datadict[domnitem]['Y']
    print('Resampling data in domain {}'.format(domnitem))
    adasyn = ADASYN(ratio = ratiodic[domnitem], random_state = 42)
    targdata, targlabl = adasyn.fit_sample(sorcdata, sorclabl)
    print('Saving data in domain {}'.format(domnitem))
    for imagcode, targimag in enumerate(targdata):
      lablcode = targlabl[imagcode]
      if lablcode not in lablcout:
        lablcout[lablcode] = 0
        lablnumb[lablcode] = 0
      else:
        lablcout[lablcode] += 1
    for imagcode, targimag in enumerate(targdata):
      lablcode = targlabl[imagcode]
      lablname = labldict[domnitem][lablcode]
      lablnumb[lablcode] += 1
      strsleng = len(str(lablcout[lablcode]))
      numbstrs = str(lablnumb[lablcode]).zfill(strsleng)
      targpath = os.path.join(savepath, domnitem, lablname)
      if not os.path.exists(targpath): os.makedirs(targpath)
      imagpath = os.path.join(targpath, 'img_' + numbstrs)
      targimag = targimag.reshape(256, 256, 3)
      cv2.imwrite(imagpath + '.jpg', targimag)

if __name__=='__main__':
  datapath = './domain_adaptation_images'
  savepath = './domain_adaptation_images_resampled'
  datadict, labldict = read_data(datapath)
  resampling(datadict, labldict, savepath)