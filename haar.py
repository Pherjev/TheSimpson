import cv2
import numpy as np
from os import listdir
import re
import imghdr

def sorted_alphanumeric(data):
        convert = lambda text: int(text) if text.isdigit() else text.lower()
        alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)',key)]
        return sorted(data,key=alphanum_key)

face_cascade = cv2.CascadeClassifier('cascade.xml')

clases = ['charles_montgomery_burns','homer_simpson','krusty_the_clown','moe_szyslak','ned_flanders','principal_skinner','bart_simpson','lisa_simpson','maggie_simpson','marge_simpson']

print(len(clases))

for clase in clases:

	directory = 'simpsons_dataset/simpsons_dataset/' + clase + '/'
	dirlist = sorted_alphanumeric(listdir(directory))

	d1 = 0
	d2 = 0
	c = 0


	for name0 in dirlist:

		name = 'simpsons_dataset/simpsons_dataset/' + clase + '/' + name0
		img = cv2.imread(name)
		tipo = imghdr.what(name)
		if tipo == 'jpg' or tipo == 'jpeg':
			#img = cv2.imread(name)
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

			faces = face_cascade.detectMultiScale(gray, 1.3,5) 


	
			for (x,y,w,h) in faces:
				c += 1
				d1 += h
				d2 += w
				img2 = img[y:y+h, x:x+w]
				img2 = cv2.resize(img2,(195,195))
				cv2.imwrite('training_set/' + clase + '/img' + str(int(c)) + '.jpg', img2)
				#cv2.imshow('img',img2)
				#cv2.waitKey(0)


	d1 = d1 / c
	d2 = d2 / c

	print(d1,d2)
