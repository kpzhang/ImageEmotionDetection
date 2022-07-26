import os,sys,re
import codecs

import cv2

from sklearn import ensemble
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing

from keras.preprocessing import image

import matplotlib.pyplot as plt

import numpy as np

seed = 2019
np.random.seed(seed)
vectorizer = CountVectorizer()
le = preprocessing.LabelEncoder()


height,width = 224,224

fh = open(sys.argv[1],'r')
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
	image_id = arr[0].strip().replace(label,'')
	anp = arr[2].strip()
	tag = arr[3].strip()

	img_fname = '../data/'+label+'/'+label+'_'+'0'*(4-len(image_id))+image_id+'.jpg'

	if len(anp.strip()) == 0 or len(tag.strip()) == 0 or not os.path.exists(img_fname):
		continue

	anps.append(anp)
	tags.append(tag)

	img = image.load_img(img_fname, target_size=(height, width,3))
	img = image.img_to_array(img)
	img = img.reshape(1,-1)[0]
	img = img/255
	img_X.append(img)
	labels.append(label)
img_X = np.array(img_X)

anp_X = vectorizer.fit_transform(anps).toarray()
tag_X = vectorizer.fit_transform(tags).toarray()
y = np.array(labels)
#print type(img_X),type(anp_X),type(tag_X)
#print img_X.shape,anp_X.shape,tag_X.shape,y.shape
print 'ready for XGBoost...'

## training
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
cvscores = []
#model = ensemble.GradientBoostingClassifier()
model = ensemble.RandomForestClassifier()

#X = img_X
X = np.hstack((img_X,anp_X,anp_X))

for train, test in kfold.split(X):
	#print X[train].shape,',',y[train].shape
	model.fit(X[train],y[train])

	score = model.score(X[test],y[test])
	print("%.2f%%" % (score*100))
	cvscores.append(score * 100)

print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
