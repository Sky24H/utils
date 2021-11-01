import subprocess
import os,cv2
import glob


def break_into_frames(video_path, saev_dir):
    cmd_1 = 'rm -rf '+ saev_dir
    subprocess.call(cmd_1, shell=True)
    os.makedirs(saev_dir, exist_ok=True)

    cmd_2 = 'ffmpeg -i ' + video_path + ' ' + saev_dir + '/%d.png'
    subprocess.call(cmd_2, shell=True)

def process_2(vid_1, vid_2, new_path):
    cmd_1 = 'rm -rf '+ '.temp_3'
    subprocess.call(cmd_1, shell=True)
    os.makedirs('.temp_3', exist_ok=True)

    break_into_frames(vid_1, '.temp_1')
    break_into_frames(vid_2, '.temp_2')
    all_imgs_1 = glob.glob('.temp_1/*.png')
    all_imgs_2 = glob.glob('.temp_2/*.png')
    for i in range(min(len(all_imgs_1), len(all_imgs_2))):
        img_1, img_2 = cv2.imread(all_imgs_1[i]), cv2.imread(all_imgs_2[i])
        if img_1.shape[0] != 256 :
            img_1 = cv2.resize(img_1, (256, 256))
        # if img_2.shape[0] != 256 or img_2.shape[1] != 256 :
        # img_2 = cv2.resize(img_2, (256, 256))
        new_frame = cv2.hconcat([img_1, img_2])
        # print(new_frame.shape)
        new_frame_path = '.temp_3/'+str(i+1)+'.png'
        cv2.imwrite(new_frame_path, new_frame)
    # cmd_1 = 'ffmpeg -i ' + vid_2 + ' ' + os.path.basename(vid_2)[:-4] + '.wav'
    # print(cmd_1)
    # subprocess.call(cmd_1, shell=True)
    cmd_2 = 'ffmpeg -y -i .temp_3/%d.png -vf fps=25 -strict -2 ' + new_path
    subprocess.call(cmd_2, shell=True)

def process_3(vid_1, vid_2, vid_3, new_path):
    process_2(vid_1, vid_2, '.test.mp4')
    process_2('.test.mp4', vid_3, new_path)
    cmd = 'rm -rf .test.mp4'
    subprocess.call(cmd, shell=True)


if __name__ == "__main__":
    vid_1 = './demo/videos/1.mp4'
    vid_2 = './result_1.mp4'
    new_path = './compare_1.mp4'
    process_2(vid_1, vid_2, new_path)