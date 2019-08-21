from PIL import Image
import PIL.ImageOps
import os
import numpy as np
import cv2

def crop(image):
    y = 17
    x = 23

    img1 = image[y: y + 18, x: x + 36]

    return img1

for curDir, dirs, files in os.walk('./0819_data'):
	for file in files:
		image = np.array(Image.open(os.path.join(curDir, file)), dtype=np.float32)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image = crop(image)
		cv2.imwrite(os.path.join('./temp', file),image,[cv2.IMWRITE_JPEG_QUALITY, 100])

# chip`s color of good images were slightly different compare with bad images`, so this method may not work, just for reference.
