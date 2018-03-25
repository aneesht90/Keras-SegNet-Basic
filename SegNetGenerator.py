from __future__ import absolute_import
from __future__ import print_function
import os

from keras import backend as K
from keras import models
from keras.optimizers import SGD, Adadelta
from keras.utils import plot_model
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint


import cv2
import numpy as np
import json

#from DataGenerator import DataGenerator
from utils.image_reader import (
    RandomTransformer,
    SegmentationDataGenerator)

np.random.seed(7) # 0bserver07 for reproducibility
data_shape = 192*256
class_weighting= [0.2595, 0.1826, 4.5640, 0.1417, 0.5051, 0.3826, 9.6446, 1.8418, 6.6823, 6.2478, 3.0, 7.3614]
nb_epoch = 10
batch_size = 3
test_network = False


# load the model:
with open('segNet_basic_model_.json') as model_file:
    segnet_basic = models.model_from_json(model_file.read())
print(segnet_basic.summary())


#segnet_basic.compile(loss="categorical_crossentropy", optimizer='adadelta', metrics=["accuracy"])
segnet_basic.compile(loss="sparse_categorical_crossentropy", optimizer='adadelta', metrics=["accuracy"])

# checkpoint
#filepath="weights.best.hdf5"
#checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False)
callbacks_list = [tensorboard]

DataPath = './CamVid/'


def load_filenames(mode):
    data = []
    label = []
    with open(DataPath + mode +'.txt') as f:
        #print("path found")
        txt = f.readlines()
        txt = [line.split(' ') for line in txt]

    # print("Test print",os.getcwd() +txt[count][0][7:])
    # print("Test print sophist",os.getcwd() +txt[count][0][7:])
    # print("Label print",os.getcwd() +txt[count][1][7:][:-1])

    data =  [os.getcwd() +txt[i][0][7:] for i in range(len(txt))]
    label =  [os.getcwd() +txt[i][1][7:][:-1] for i in range(len(txt))]
    print('.',end='')
    return np.array(data), np.array(label)


train_img_fnames, train_mask_fnames = load_filenames("train")
val_img_fnames, val_mask_fnames     = load_filenames("val")


# reduce data set for testing
if test_network==True:
    train_img_fnames = train_img_fnames[:3]
    train_mask_fnames = train_mask_fnames[:3]
    val_img_fnames = val_img_fnames[:3]
    val_mask_fnames = val_mask_fnames[:3]


# Printing filename lists
# print("file names",train_img_fnames)
# print("length of base names",len(train_img_fnames))


# Generators

transformer_train = RandomTransformer(horizontal_flip=True, vertical_flip=True)
datagen_train     = SegmentationDataGenerator(transformer_train)

transformer_val   = RandomTransformer(horizontal_flip=False, vertical_flip=False)
datagen_val       = SegmentationDataGenerator(transformer_val)



#print()
segnet_basic.fit_generator(
        datagen_train.flow_from_list(
            train_img_fnames,
            train_mask_fnames,
            shuffle=True,
            batch_size=batch_size,
            img_target_size=(192, 256),
            mask_target_size=(192, 256)),
            samples_per_epoch=len(train_img_fnames)//batch_size,
            nb_epoch=nb_epoch,
            validation_data=datagen_val.flow_from_list(
                val_img_fnames,
                val_mask_fnames,
                batch_size=batch_size,
                img_target_size=(192, 256),
                mask_target_size=(192, 256)),
                nb_val_samples=len(val_img_fnames),
                callbacks=callbacks_list)

# This save the trained model weights to this file with number of epochs
segnet_basic.save_weights('weights/model_weight_{}.hdf5'.format(nb_epoch))
