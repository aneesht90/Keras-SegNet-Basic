from __future__ import absolute_import
from __future__ import print_function
from tqdm import tqdm
import cv2
import numpy as np
import itertools
from os import listdir
from os.path import isfile, join
from helper import *
import os
import glob
from PIL import Image


# Copy the data to this dir here in the SegNet project /CamVid from here:
# https://github.com/alexgkendall/SegNet-Tutorial
DataPath = './CamVid/'
image_height  = 360   # {192,360}
image_width   = 480   # {256,480}
segment_count = 12
train_count   = 5000
data_shape    = image_height * image_width
label_mode    = 'one_hot'   # {'sparse','one_hot'}       # label in the pixel stored with additional dimension
                                                        # as one hot or sparse integer representation
softmax_mode  = 'original' # {'original','flatten'}     # network output flattened or without modification
augmented     = True       #{True, False}

def load_data(mode):
    data = []
    label = []
    with open(DataPath + mode +'.txt') as f:
        #print("path found")
        txt = f.readlines()
        txt = [line.split(' ') for line in txt]
    for i in range(len(txt)):
        #data.append(np.rollaxis(normalized(cv2.imread(os.getcwd() + txt[i][0][7:])),2))
        #data.append(np.rollaxis(normalized(cv2.resize(cv2.imread(os.getcwd() + txt[i][0][7:])),dsize=(image_width, image_height)),0))

        data.append(np.rollaxis(normalized(cv2.resize(cv2.imread(os.getcwd() + txt[i][0][7:]),dsize=(image_width,image_height))),0))

        if label_mode == 'one_hot':
            label.append(one_hot_it(cv2.resize(cv2.imread(os.getcwd() + txt[i][1][7:][:-1]),dsize=(image_width,image_height) )[:,:,0]))
        elif label_mode == 'sparse':
            label.append(cv2.resize(cv2.imread(os.getcwd() + txt[i][1][7:][:-1],cv2.IMREAD_GRAYSCALE),dsize=(image_width,image_height)))
        print('.',end='')
    return np.array(data), np.array(label)


def load_data_augmented(directory):
    data  = []
    label = []
    base_dir   = os.path.join(os.getcwd(),directory)
    dir_images = os.path.join(base_dir,'images')
    dir_masks  = os.path.join(base_dir,'masks')
    #glob.glob("/home/adam/*.txt")
    filelist  = [f for f in listdir(dir_images) if isfile(join(dir_images, f))]
    file_count = 0
    progress_bar = tqdm(total=train_count, desc="Reading data", unit=' Samples', leave=False)
    for fname in filelist:
        if (isfile(join(dir_images, fname)) and isfile(join(dir_images, fname)) ):
            #data.append(np.rollaxis(normalized(cv2.imread(join(dir_images, fname))),0))
            if label_mode == 'one_hot':
                #label.append(one_hot_it( cv2.imread(join(dir_masks, fname)) )[:,:,0]))
                label.append(one_hot_it(cv2.imread(join(dir_masks, fname))[:,:,0]))
            elif label_mode == 'sparse':
                label.append(cv2.imread( (join(dir_masks, fname)),cv2.IMREAD_GRAYSCALE))
        else:
            print("file {} does not exist".format(fname))
        progress_bar.set_description("Processing ")
        progress_bar.update(1)
    progress_bar.close()
    return np.array(data), np.array(label)


if augmented==True:
    train_data, train_label = load_data_augmented('CamVid/train/output')
    #print("type of train labels: ",train_label)
    #np.save("data_aug/train_data", train_data)
    if softmax_mode == 'original':
        if label_mode == 'one_hot':
            train_label = np.reshape(train_label,(train_count,image_height,image_width,segment_count))
        elif label_mode == 'sparse':
            train_label = np.reshape(train_label,(train_count,image_height,image_width,1))
        np.save("data_aug/train_label", train_label)
    elif softmax_mode == 'flatten':
        if label_mode == 'one_hot':
            train_label = np.reshape(train_label,(train_count,data_shape,segment_count))
        elif label_mode == 'sparse':
            train_label = np.reshape(train_label,(train_count,data_shape,1))
        np.save("data_aug/train_label", train_label)


else:
    # Training data
    # train_data, train_label = load_data("train")
    # np.save("data/train_data", train_data)
    #
    # if softmax_mode == 'original':
    #     if label_mode == 'one_hot':
    #         train_label = np.reshape(train_label,(367,image_height,image_width,segment_count))
    #     elif label_mode == 'sparse':
    #         train_label = np.reshape(train_label,(367,image_height,image_width,1))
    #     np.save("data/train_label", train_label)
    # elif softmax_mode == 'flatten':
    #     if label_mode == 'one_hot':
    #         train_label = np.reshape(train_label,(367,data_shape,segment_count))
    #     elif label_mode == 'sparse':
    #         train_label = np.reshape(train_label,(367,data_shape,1))
    #     np.save("data_/train_label", train_label)


    # Test data
    test_data, test_label = load_data("test")
    np.save("data/test_data", test_data)
    print("shape of test data",test_data.shape)
    print("shape of test label",test_label.shape)
    if softmax_mode == 'original':
        if label_mode == 'one_hot':
            test_label = np.reshape(test_label,(233,image_height,image_width,segment_count))
        elif label_mode == 'sparse':
            test_label = np.reshape(test_label,(233,image_height,image_width,1))
        np.save("data/test_label", test_label)
    elif softmax_mode == 'flatten':
        if label_mode == 'one_hot':
            test_label = np.reshape(test_label,(233,data_shape,segment_count))
        elif label_mode == 'sparse':
            test_label = np.reshape(test_label,(233,data_shape,1))
        np.save("data_/test_label", test_label)




    # Validation data
    val_data, val_label = load_data("val")
    np.save("data/val_data", val_data)
    if softmax_mode == 'original':
        if label_mode == 'one_hot':
            val_label = np.reshape(val_label,(101,image_height,image_width,segment_count))
        elif label_mode == 'sparse':
            val_label = np.reshape(val_label,(101,image_height,image_width,1))
        np.save("data/val_label", val_label)
    elif softmax_mode == 'flatten':
        if label_mode == 'one_hot':
            val_label = np.reshape(val_label,(101,data_shape,segment_count))
        elif label_mode == 'sparse':
            val_label = np.reshape(val_label,(101,data_shape,1))
        np.save("data_/val_label", val_label)



# FYI they are:
# Sky = [128,128,128]
# Building = [128,0,0]
# Pole = [192,192,128]
# Road_marking = [255,69,0]
# Road = [128,64,128]
# Pavement = [60,40,222]
# Tree = [128,128,0]
# SignSymbol = [192,128,128]
# Fence = [64,64,128]
# Car = [64,0,128]
# Pedestrian = [64,64,0]
# Bicyclist = [0,128,192]
# Unlabelled = [0,0,0]
