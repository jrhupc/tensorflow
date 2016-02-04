#/usr/local/bin/python

import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from IPython.display import display, Image
from scipy import ndimage
from sklearn import linear_model
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle


# If needed re-read pickle
pickle_file = 'notMNIST.pickle'
d = pickle.load(open(pickle_file,'rb'))

print type(d)
print type(d['train_labels'])
print d['train_labels'].shape


train_dataset = d['train_dataset']
train_labels = d['train_labels']
valid_dataset = d['valid_dataset']
valid_labels = d['valid_labels']
test_dataset = d['test_dataset']
test_labels = d['test_labels']

## # Check if there is a similar image (ssim>0.95) in the validation dataset\n",
## # That is just too slow... \n",
## #from skimage.measure import structural_similarity as ssim\n",
## thres = 0.01

## for i,im1 in enumerate(train_dataset):
##     #print i
##     for j,im2 in enumerate(valid_dataset):
##         #if (ssim(im1,im2) > thres):
##         if np.sum(np.absolute(im1 - im2)) < thres:
##             print "train {} also as valid {}".format(i,j)

print type(train_dataset)

print train_dataset.shape

#logreg = linear_model.LogisticRegression()
#logreg.fit(train_dataset,train_labels)
#print('Variance score: %.2f' % logreg.score(valid_dataset, valid_labels))

