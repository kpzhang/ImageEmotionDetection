import os,sys,re
import time
import codecs

import cv2

from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing

import scipy
from keras.preprocessing import image

import numpy as np
import tensorflow as tf

seed = 2019
np.random.seed(seed)
vectorizer = CountVectorizer()
le = preprocessing.LabelEncoder()


height,width = 224,224

fh = open('combined_data_7607.csv','r')
lines = fh.readlines()
fh.close()

img_X = []
y = []
labels = []


anps = []
tags = []
for i in range(len(lines)):
	arr = lines[i].strip().split(',')
	label = arr[1].strip()
	if label == '':
		continue
	if label == 'sad':
		label = 'sadness'
	image_id = arr[0].strip()
	anp = arr[2].strip()
	tag = arr[3].strip()

	img_fname = './data/'+image_id+'.jpg'

	if len(anp.strip()) == 0 or len(tag.strip()) == 0 or not os.path.exists(img_fname):
		continue

	anps.append(anp)
	tags.append(tag)

	img = image.load_img(img_fname, target_size=(height, width,3))
	img = image.img_to_array(img)
	img = img/255
	img_X.append(img)
	labels.append(label)
img_X = np.array(img_X)


le.fit(labels)
emotions = le.classes_
n = len(emotions)
index = le.transform(labels)
for i in range(len(index)):
	ind = index[i]
	encoding = [0]*n
	encoding[ind] = 1
	y.append(encoding)

anp_X = vectorizer.fit_transform(anps)
tag_X = vectorizer.fit_transform(tags)
y = np.array(y)
#print(img_X.shape, anp_X.shape, tag_X.shape, y.shape)
print 'ready for deep learning...'

## training
from keras import backend as K
from keras.applications import VGG16, ResNet50, Xception, NASNetLarge

from keras import optimizers
from keras.layers import Dense, Input
from keras.models import Model
from keras.layers.merge import concatenate


img_input = Input(shape=(224,224,3),name='image_input')
#net = VGG16(weights='imagenet', include_top=True)
net = Xception(weights='imagenet', include_top=True)
#net = NASNetLarge(weights='imagenet', include_top=True)
net.trainable = False
for l in net.layers:
	l.trainable = False
img_out = net(img_input)
img_out = Dense(256,activation='relu',name='img_out')(img_out)

anp_input = Input(shape=(anp_X.shape[1],),name='anp_input')
anp_out = Dense(256,activation='relu',name='anp_out')(anp_input)

tag_input = Input(shape=(tag_X.shape[1],),name='tag_input')
tag_out = Dense(256,activation='relu',name='tag_out')(tag_input)

all_out = concatenate([img_out,anp_out,tag_out],name='alltogether')
all_out = Dense(256,activation='relu')(all_out)
prediction = Dense(n,activation='softmax',name='output')(all_out)

model = Model(inputs=[img_input,anp_input,tag_input],outputs=prediction)
model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
#print(model.summary())

BATCH_SIZE = 64
NB_EPOCH = 20
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
cvscores = []
t1 = time.time()
for train, test in kfold.split(img_X):
	#train,test = scipy.sparse.csr_matrix.sort_indices(train), scipy.sparse.csr_matrix.sort_indices(test)
	model.fit([img_X[train], anp_X[train],tag_X[train]],y[train], batch_size=BATCH_SIZE, epochs=NB_EPOCH, verbose=0)

	scores = model.evaluate([img_X[test], anp_X[test],tag_X[test]],y[test], verbose=0)
	print "%s: %.2f%%" % (model.metrics_names[1], scores[1]*100)
	cvscores.append(scores[1] * 100)
	print "time:",time.time() - t1
	t1 = time.time()

print "%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores))
