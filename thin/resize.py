import os
import cv2
a = './data_new/trainB/'
for root,dirs,files in os.walk(a):
	for name in files:
		#print(a+name)
		pic = cv2.imread(a+name)
		pic = cv2.resize(pic, (512, 512), interpolation=cv2.INTER_CUBIC)
		cv2.imwrite('./data_new/trainA/'+name,pic,[cv2.IMWRITE_JPEG_QUALITY, 100])

