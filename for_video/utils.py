import numpy as np
import cv2, os, sys, glob
import face_detection
import face_alignment
from tqdm import tqdm
from natsort import natsorted
device = "cuda"
import shutil
import torch



landmarks_68_pt = { "mouth": (48,68),
                    "right_eyebrow": (17, 22),
                    "left_eyebrow": (22, 27),
                    "right_eye": (36, 42),
                    "left_eye": (42, 48),
                    "nose": (27, 36), # missed one point
                    "jaw": (0, 17) }


def face_detect_landmarks(images):

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device=device)
    batch_size = 32

    while 1:
        landmarks = []
        try:
            for i in range(0, len(images), batch_size):
                image_batch = np.array(images[i:i + batch_size])
                landmarks.extend(fa.get_landmarks_from_batch(torch.Tensor(image_batch.transpose(0, 3, 1, 2))))

        except RuntimeError:
            if batch_size == 1:
                raise RuntimeError(
                    'Image too big to run face detection on GPU. Please use the --resize_factor argument')
            batch_size //= 2
            print('Recovering from OOM error; New batch size: {}'.format(batch_size))
            continue
        break
    # landmarks = np.array(landmarks) - (rect[0], rect[1])
    del fa
    return np.array(landmarks)


def face_detect_rect_first_one_only(images_ori):
    nosmooth = False
    pads = [0, 0, 0, 0]
    face_det_batch_size = 1
    images = images_ori[:1]
    detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D,
                                            flip_input=False, device=device)
    batch_size = face_det_batch_size

    while 1:
        predictions = []
        try:
            for i in range(0, len(images), batch_size):
                predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
        except RuntimeError:
            if batch_size == 1:
                raise RuntimeError(
                    'Image too big to run face detection on GPU. Please use the --resize_factor argument')
            batch_size //= 2
            print('Recovering from OOM error; New batch size: {}'.format(batch_size))
            continue
        break

    results = []
    pady1, pady2, padx1, padx2 = pads
    for rect, image in zip(predictions, images):
        if rect is None:
            # check this frame where the face was not detected.
            # cv2.imwrite('temp/faulty_frame.jpg', image)
            print('Face not detected! Ensure the video contains a face in all the frames.')
            return []

        y1 = max(0, rect[1] - pady1)
        y2 = min(image.shape[0], rect[3] + pady2)
        x1 = max(0, rect[0] - padx1)
        x2 = min(image.shape[1], rect[2] + padx2)
        # y_gap, x_gap = (y2 - y1)//2, (x2 - x1)//2
        y_gap = min((y2 - y1)//2, y1, image.shape[0] - y2)
        x_gap = (((y2-y1)+y_gap*2) - (x2-x1))//2
        coords = y1-y_gap, y2+y_gap, x1-x_gap, x2+x_gap
        #print(coords)
        # coords = [coords_[0], coords_[0]+1024, coords_[2], coords_[2]+1024]
        #results.append(image[y1-y_gap: y2+y_gap, x1-x_gap:x2+x_gap])
    results = [image[coords[0]:coords[1], coords[2]:coords[3]] for image in images_ori]
    del detector
    return results



def draw_landmarks (image, image_landmarks, color=(0,255,0), draw_circles=True, thickness=1, transparent_mask=False, only_mouth=True):
    if len(image_landmarks) != 68:
        raise Exception('get_image_eye_mask works only with 68 landmarks')

    int_lmrks = np.array(image_landmarks, dtype=np.int)
    mouth = int_lmrks[slice(*landmarks_68_pt["mouth"])]
    print(mouth)

    if only_mouth:
        cv2.polylines(image, tuple(np.array([v]) for v in (mouth,)),
                  True, color, thickness=thickness, lineType=cv2.LINE_AA)
    else:
        # right_eye = int_lmrks[slice(*landmarks_68_pt["right_eye"])]
        # left_eye = int_lmrks[slice(*landmarks_68_pt["left_eye"])]
        # nose = int_lmrks[slice(*landmarks_68_pt["nose"])]
        # jaw = int_lmrks[slice(*landmarks_68_pt["jaw"])]
        # right_eyebrow = int_lmrks[slice(*landmarks_68_pt["right_eyebrow"])]
        # left_eyebrow = int_lmrks[slice(*landmarks_68_pt["left_eyebrow"])]
        # open shapes
        cv2.polylines(image, tuple(np.array([v]) for v in ( right_eyebrow, jaw, left_eyebrow, np.concatenate((nose, [nose[-6]])) )),
                    False, color, thickness=thickness, lineType=cv2.LINE_AA)
        # closed shapes
        cv2.polylines(image, tuple(np.array([v]) for v in (right_eye, left_eye, mouth)),
                    True, color, thickness=thickness, lineType=cv2.LINE_AA)


def face_detect_rect(images, images_ori):
    nosmooth = False
    pads = [0, 0, 0, 0]
    face_det_batch_size = 32
    detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D,
                                            flip_input=False, device=device)
    batch_size = face_det_batch_size

    while 1:
        predictions = []
        try:
            for i in range(0, len(images), batch_size):
                predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
        except RuntimeError:
            if batch_size == 1:
                raise RuntimeError(
                    'Image too big to run face detection on GPU. Please use the --resize_factor argument')
            batch_size //= 2
            print('Recovering from OOM error; New batch size: {}'.format(batch_size))
            continue
        break

    all_coords = []
    results = []
    pady1, pady2, padx1, padx2 = pads
    for rect, image in zip(predictions, images):
        if rect is None:
            # check this frame where the face was not detected.
            cv2.imwrite('temp/faulty_frame.jpg', image)
            print('Face not detected! Ensure the video contains a face in all the frames.')
            return []

        y1 = max(0, rect[1] - pady1)
        y2 = min(image.shape[0], rect[3] + pady2)
        x1 = max(0, rect[0] - padx1)
        x2 = min(image.shape[1], rect[2] + padx2)
        y_gap, x_gap = (y2 - y1)//6, (x2 - x1)//6
        all_coords.append(np.array([y1, y2, x1, x2]))
        results.append(image[y1-y_gap: y2+y_gap, x1-x_gap:x2+x_gap])
    var = np.var(all_coords, axis=0)
    del detector

    if sum(var) > 100:
        results = [cv2.resize(img, (128,128)) for img in results]
    else:
        first_rect = predictions[0]

        y1 = max(0, rect[1] - pady1)
        y2 = min(image.shape[0], rect[3] + pady2)
        x1 = max(0, rect[0] - padx1)
        x2 = min(image.shape[1], rect[2] + padx2)
        y_gap, x_gap = (y2 - y1)//6, (x2 - x1)//6
        results = [cv2.resize(img[y1-y_gap: y2+y_gap, x1-x_gap:x2+x_gap], (128,128)) for img in images]
    return results


def get_faces(data_name):
    data_dir = os.path.join(data_name, 'raw_frames_audio')
    dirs = glob.glob(os.path.join(data_dir,"*"))
    dirs.sort()

    save_dir = os.path.join(data_name, 'processed_dataset')
    os.makedirs(save_dir, exist_ok=True)

    for dir_ in tqdm(dirs):
            current_dir = dir_.replace('raw_frames_audio', 'processed_dataset')
            os.makedirs(current_dir, exist_ok=True)
            # print(current_dir)
            imgs = [cv2.imread(path) for path in natsorted(glob.glob(dir_+"/*.png"))] + [cv2.imread(path) for path in natsorted(glob.glob(dir_+"/*.jpg"))]
            results = face_detect_rect(imgs)
            from_wav = dir_ + '/audio.wav'
            shutil.copyfile(from_wav, os.path.join(current_dir, 'audio.wav'))
            for i, res in enumerate(results):
                #print(i, res.shape)
                if res.shape[0] != 0 and res.shape[1] != 0:
                    cv2.imwrite(os.path.join(current_dir, str(i+1)+'.jpg'), cv2.resize(res, (128,128)))


def check_log(data_name):
    log_file = os.path.join('./logs', data_name+'.log')
    if os.path.exists(log_file):
        with open(log_file, 'r') as r:
            log = r.read().splitlines()
        if len(log) > 0:
            print(log)
            finished = log[-1].split(',')[0][-1]
            return int(finished)
        else:
            return 0
    else:
        return 0


if __name__ == '__main__':
    pass
    # get_faces('kiriyama_0623')
