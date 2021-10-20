import os
import cv2


def rename(path, save_path):
    count = 0
    for root, dirs, files in os.walk(path):
        for name in files:
            # print(a+name)
            count += 1
            pic = cv2.imread(a+name)
            #pic = cv2.resize(pic, (80, 80), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(save_path+str(count)+".png",
                        pic, [cv2.IMWRITE_JPEG_QUALITY, 100])


path = './masks/'
save_path = './mask_processed/'
rename(path, save_path)
