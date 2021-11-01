from PIL import Image
import PIL.ImageOps
import os
import numpy as np
import cv2


def get_random_crop(image, crop_height, crop_width, flag):
    y = 15

    if flag == 0:
        x = 55
    else:
        x = 10

    abnormal = image[y: y + crop_height, x: x + crop_width]

    return abnormal


data_dir = './left'

for curDir, dirs, files in os.walk(data_dir):
    for file in files:
        image = np.array(Image.open(
            os.path.join(curDir, file)), dtype=np.float32)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = get_random_crop(image, 20, 20, 1)
        cv2.imwrite(os.path.join('./left_test', file),
                    image, [cv2.IMWRITE_JPEG_QUALITY, 100])
