from PIL import Image
import PIL.ImageOps
import os
import numpy as np
import cv2

# For object in center and backgound is close to balck or white.


def remove_back(path, save_path):
    for curDir, dirs, files in os.walk(path):
        for file in files:
            image = np.array(Image.open(
                os.path.join(curDir, file)), dtype=np.float32)
            gray = np.array(Image.open(os.path.join(curDir, file)
                                       ).convert('L'), dtype=np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            for i in range(20):
                for j in range(20):
                    if gray[i, j] < 45:
                        image[i, j] = 0
            cv2.imwrite(os.path.join(save_path, file), image,
                        [cv2.IMWRITE_JPEG_QUALITY, 100])


path = './mask'
save_path = './lala'
remove_back(path, save_path)
