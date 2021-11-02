import numpy as np
import cv2, os, sys, glob
from tqdm import tqdm
from natsort import natsorted
device = "cuda"
import shutil
from utils import face_detect_rect_first_one_only as face_detect_rect
from utils import draw_landmarks
from utils import face_detect_landmarks


def get_faces(data_dir_1, data_dir_2):
    dirs = glob.glob(os.path.join(data_dir_1,"*"))
    dirs.sort()
    os.makedirs(data_dir_2, exist_ok=True)
    for dir_ in tqdm(dirs):
        current_dir = os.path.join(data_dir_2, os.path.basename(dir_))
        os.makedirs(current_dir, exist_ok=True)
        print('processing', current_dir)
        imgs = [cv2.imread(path) for path in natsorted(glob.glob(dir_+"/*.png"))] + [cv2.imread(path) for path in natsorted(glob.glob(dir_+"/*.jpg"))]
        results = face_detect_rect(imgs)

        for i, res in enumerate(results):
            #print(i, res.shape)
            if res.shape[0] != 0 and res.shape[1] != 0:
                cv2.imwrite(os.path.join(current_dir, str(i+1)+'.png'), cv2.resize(res, (256,256)))

data_dir_1 = '/mnt/data/huang/datasets/faces/raw_frames_audio_10'
data_dir_2 = '/mnt/data/huang/datasets/faces/datasets_processed'

get_faces(data_dir_1, data_dir_2)
