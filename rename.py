import os
import subprocess
import glob

root = './data_0118/'
dir_list = os.listdir(root)
print(dir_list)
for dir_ in dir_list:
    folder_list = glob.glob(root+dir_+'/*')
    for f in folder_list:
        cmd = 'mv ' + f + ' ' + f.replace('/0', '/old_0')
        subprocess.call(cmd, shell=True)