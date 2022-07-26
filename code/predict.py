import os,sys,re
import codecs

import cv2

from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing

from keras.preprocessing import image
from keras import backend as K
from keras.applications import Xception

from keras import optimizers
from keras.layers import Dense, Input, Flatten
from keras.models import Model
from keras.layers.merge import concatenate

import matplotlib.pyplot as plt

import numpy as np

seed = 2019
np.random.seed(seed)
vectorizer = CountVectorizer()
le = preprocessing.LabelEncoder()


height,width = 224,224

fh = open(sys.argv[1],'r') # combined_data_7607.csv
lines = fh.readlines()
fh.close()

img_X = []
y = []
labels = []

anps = []
tags = []
for i in range(1,len(lines)):
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
img_X_train = np.array(img_X)

le.fit(labels)
emotions = le.classes_
n = len(emotions)
index = le.transform(labels)
for i in range(len(index)):
	ind = index[i]
	encoding = [0]*n
	encoding[ind] = 1
	y.append(encoding)

# get the ANPs
fh = open('ANPs.csv','r')
lines = fh.readlines()
fh.close()
anps_mapping = {}
for i in range(len(lines)):
	arr = lines[i].strip().split(',')
	img_id = arr[0].strip()
	anps_mapping[img_id] = ' '.join([k.replace('_',' ') for k in arr[1:]])


# get the tags
fh = open('tags.txt','r')
lines = fh.readlines()
fh.close()
tags_mapping = {}
for i in range(len(lines)):
	arr = lines[i].strip().split(',')
	img_id = arr[0].strip()
	tags_mapping[img_id] = ' '.join([k.replace('_',' ') for k in arr[1:]])

img_ids = []
for img_id in tags_mapping:
	img_ids.append(img_id)
	anps.append(anps_mapping[img_id])
	tags.append(tags_mapping[img_id])

anp_X= vectorizer.fit_transform(anps)
tag_X = vectorizer.fit_transform(tags)
print img_X_train.shape,anp_X.shape,tag_X.shape
y = np.array(y)

# obtain features for testing images
img_X_test = []
for i in range(len(img_ids)):
	img_id = img_ids[i]
	img_fname = 'kickstarter_images/'+img_id+'.png'
	img = image.load_img(img_fname, target_size=(height, width,3))
	img = image.img_to_array(img)
	img = img/255
	img_X_test.append(img)
img_X_test = np.array(img_X_test)


## training process
BATCH_SIZE = 64
NB_EPOCH = 20
n_training = img_X_train.shape[0]

img_input = Input(shape=(224,224,3),name='image_input')
net = Xception(weights='imagenet', include_top=True)
net.trainable = False
for l in net.layers:
	l.trainable = False
img_out = net(img_input)
img_out = Dense(256,activation='relu',name='img_out')(img_out)

anp_input = Input(shape=(anp_X.shape[1],),name='anp_input')
anp_out = Dense(256,activation='relu',name='anp_out')(anp_input)

tag_input = Input(shape=(tag_X.shape[1],),name='tag_input')
tag_out = Dense(256,activation='relu',name='tag_out')(tag_input)

all_out = concatenate([img_out,anp_out,tag_out])
all_out = Dense(256,activation='relu')(all_out)
prediction = Dense(n,activation='softmax')(all_out)

model = Model(inputs=[img_input,anp_input,tag_input],outputs=prediction)
model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])

model.fit([img_X_train,anp_X[:n_training],tag_X[:n_training]],y, batch_size=BATCH_SIZE, epochs=NB_EPOCH, verbose=0)

output = model.predict([img_X_test, anp_X[n_training:],tag_X[n_training:]], verbose=0)

# output
for i in range(len(img_ids)):
	print img_ids[i],':',','.join([str(v) for v in output[i]]),',',emotions[np.argmax(output[i])]
