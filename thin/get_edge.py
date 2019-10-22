from PIL import Image,ImageFilter 
import numpy as np
import os
import cv2 as cv

i = 0

min_size = 1666
for curDir, dirs, files in os.walk('./hed_photo'):
    for file in files:
        i += 1

        img = cv.imread(os.path.join(curDir, file),0)
        #img = cv.fastNlMeansDenoising(img,None,7,21)
        ret,edges = cv.threshold(img,140,255,cv.THRESH_BINARY)
        kernel = np.ones((1,1),np.uint8)
        edges = cv.erode(edges, kernel,iterations = 1)
        #edges = cv.Canny(img,200,255)
        edges = (255-edges)
        nb_components, output, stats, centroids = cv.connectedComponentsWithStats(edges, connectivity=8)
        sizes = stats[1:, -1]; nb_components = nb_components - 1

        img2 = np.zeros((output.shape))
#for every component in the image, you keep it only if it's above min_size
        min_size = nb_components 
        for i in range(0, nb_components):
            if sizes[i] < min_size:
                img2[output == i + 1] = 255
       
        cv.imwrite(os.path.join('./edge', file),255-(edges-img2),[cv.IMWRITE_JPEG_QUALITY, 100])
