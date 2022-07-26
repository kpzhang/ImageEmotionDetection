import os,sys,io
import json

from django.utils.encoding import smart_str, smart_unicode

from google.cloud import vision

client = vision.ImageAnnotatorClient()



def get_labels(img,n_objs):
	with io.open(img,'rb') as image_file:
		content = image_file.read()

	image = vision.types.Image(content = content)
	response = client.label_detection(image=image)
	labels = response.label_annotations

	objects = []

	for label in labels:
		objects.append(smart_str(label.description))

	return objects



for fname in os.listdir('kickstarter_data'):
	img_id = os.path.splitext(fname)[0]
	img_fname = 'kickstarter_data/'+fname
			
	objects = get_labels(img_fname,10)

	if len(objects) > 0:
		print img_id,',',','.join(objects)
