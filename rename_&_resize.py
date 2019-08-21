import os
import cv2
a = './lala/'
count = 0
for root,dirs,files in os.walk(a):
	for name in files:
		#print(a+name)
		count += 1
		pic = cv2.imread(a+name)
		#pic = cv2.resize(pic, (80, 80), interpolation=cv2.INTER_CUBIC)
		cv2.imwrite('./mask_processed/'+str(count)+".png",pic,[cv2.IMWRITE_JPEG_QUALITY, 100])

