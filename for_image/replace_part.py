from PIL import Image
import PIL.ImageOps
import os
import numpy as np
import cv2

def replace(image, ng, flag):
	y = 15

	if flag == 0:
		x = 55    
	else:
		x = 5    
	for i in range(20):
		for j in range(20):
			if ng[i,j][0] == ng[i,j][1] == ng[i,j][2] == 0:
				pass
			else:
				image[y+i,x+j][0] = ng[i,j][0]
				image[y+i,x+j][1] = ng[i,j][1]
				image[y+i,x+j][2] = ng[i,j][2]
	#blur = cv2.GaussianBlur(image[y-5:y+25,x-5:x+25],(3,3),0)
	#image[y-5:y+25,x-5:x+25] = blur
	image1 = cv2.GaussianBlur(image,(3,3),0)
	image2 = image
	image2[y-2:y+22,x-2:x+22] = 255
	return image1, image2

for curDir, dirs, files in os.walk('./0819_data'):
	for file in files:
		mask = np.array(Image.open("imgs_processed/"+str(np.random.randint(1,37))+'.png'), dtype=np.float32)
		mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
		image = np.array(Image.open(os.path.join(curDir, file)), dtype=np.float32)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		flag = np.random.randint(0,2)
		img1,img2 = replace(image,mask,flag)
		cv2.imwrite(os.path.join('./a', file),img1,[cv2.IMWRITE_JPEG_QUALITY, 100])
		cv2.imwrite(os.path.join('./b', file),img2,[cv2.IMWRITE_JPEG_QUALITY, 100])
