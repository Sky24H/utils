import os
import cv2
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras import layers
import sys
from PIL import *
import random

train_path = 'train'  # path of dataset
test_path = 'test'
# size = 25000                      #size
train_size = len(os.listdir(train_path))
test_size = len(os.listdir(test_path))
trainnames = ['0']*train_size
testnames = ['0']*test_size
#labels = [0]*size
output_train = './tfrecord/train.tfrecord'
output_test = './tfrecord/test.tfrecord'


def getlist(path, filenames):
    n = 0
    for root, dirs, files in os.walk(path):
        for name in files:
            if '.png' in name:
                filenames[n] = os.path.join(root, name)
                n = n+1
           # if 'cat' in name:
           #   labels[n] = 1
            if n % 1000 == 0:
                print('processing '+str(n)+'/'+str(len(filenames)))
# for i in range(10):
    # print(filenames[i]+'==='+str(labels[i]))
         # class List():
         #   pass                             #if need output labels
         # result = List()
         # result.labels = labels
    return filenames  # use code above & change here to result


def image_to_tfexample(images, labels):
    return tf.train.Example(features=tf.train.Features(feature={
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[images])),
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[labels])),
    }))
# https://blog.csdn.net/weixin_42001089/article/details/81172028


def gen_tfrecord(filenames, output_filename):
    with tf.Session() as sess:
        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
            for i, filename in enumerate(filenames):
                try:
                    sys.stdout.write('\r>> Converting image %d/%d' %
                                     (i+1, len(filenames)))
                    sys.stdout.flush()

                    image_data = Image.open(filename)
                    image_data = image_data.resize((256, 256))
                    image_data = np.array(image_data.convert('RGB'))
                    image_data = image_data.tobytes()
                    label = 0
                    if 'real' in filename:
                        label = 1
                    print('label is '+str(label))  # test
                    example = image_to_tfexample(image_data, label)
                    tfrecord_writer.write(example.SerializeToString())

                except IOError as e:
                    print('Could not read:', filename)
                    print('Error:', e)
    sys.stdout.write('\n')
    sys.stdout.flush()


trainnames = getlist(train_path, trainnames)
testnames = getlist(test_path, testnames)

random.shuffle(trainnames)
random.shuffle(testnames)

gen_tfrecord(trainnames, output_train)
print('----------------------trainset finished----------------------'*100)
gen_tfrecord(testnames, output_test)
