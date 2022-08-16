import os,sys

names = []
for fname in os.listdir('buffy_images'):
	names.append("buffy_images\\"+fname)
	
for i in range(int(sys.argv[1]),int(sys.argv[2])):
	print(names[i])