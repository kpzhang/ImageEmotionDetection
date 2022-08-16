import os,sys
import json
import itertools

img_ids = []
with open('buffy_path_file.txt','r') as fh:
	lines = fh.readlines()
	for i in range(len(lines)):
		img_ids.append(lines[i].strip()[:-4].replace('buffy_images\\',''))

with open('buffy_path_file.json', 'r') as f:
    data = json.load(f)

list = []
list_new = []
d = [[] for x in range(0,len(data['images']))]
for i in range(0,len(data['images'])):
	for key, value in itertools.islice(data['images'][i]['bi-concepts'].items(), 0, 10):
		d[i].append(key)

for i in range(0,len(data['images'])):
	list_new.append(",".join(d[i]))

for i in range(len(list_new)):
	print img_ids[i],',',list_new[i]	
