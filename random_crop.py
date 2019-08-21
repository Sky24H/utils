from PIL import Image
import PIL.ImageOps
import os
import numpy as np
import scipy.misc
import matplotlib
import cv2

i = 0

count = 0

def get_random_crop(image, crop_height, crop_width, flag):
    a = image.shape[1]
    b = image.shape[0]

    crop_height = np.random.randint(15,crop_height)
    crop_width = np.random.randint(20,crop_width) 

    max_x = image.shape[1] - crop_width - a/2
    max_y = image.shape[0] - crop_height - b/4
    
    x = np.random.randint(image.shape[1]/16, max_x)
    y = np.random.randint(image.shape[0]/16, max_y)
    if flag == 1:
         x = int(x*4)
    image[y: y + crop_height, x: x + crop_width] = 255


    return image


for curDir, dirs, files in os.walk('./data'):
	for file in files:
		count += 1

		image = np.array(Image.open(os.path.join(curDir, file)), dtype=np.float32)
		if count % 2 == 0:
			image = get_random_crop(image,image.shape[1]/3,image.shape[0]/2,0)
			#img = Image.fromarray(image)
			#scipy.misc.toimage(img, cmin=0.0, cmax=...).save(os.path.join('./test', file))
		else:
			image = get_random_crop(image,image.shape[1]/3,image.shape[0]/2,1)
		cv2.imwrite(os.path.join('./test', file),image)
