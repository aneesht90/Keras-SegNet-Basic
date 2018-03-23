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

tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0,
                              write_graph=True, write_images=False)
np.random.seed(7) # 0bserver07 for reproducibility
data_shape = 360*480
class_weighting= [0.2595, 0.1826, 4.5640, 0.1417, 0.5051, 0.3826, 9.6446, 1.8418, 6.6823, 6.2478, 3.0, 7.3614]
nb_epoch = 10
batch_size = 3
# Parameters
params = {'dim_x': 256,
          'dim_y': 256,
          'dim_z': 256,
          'batch_size': 3,
          'shuffle': True}

# # Datasets
# partition = # IDs
# labels = # Labels
#




# load the model:
with open('segNet_basic_model.json') as model_file:
    segnet_basic = models.model_from_json(model_file.read())


#segnet_basic.compile(loss="categorical_crossentropy", optimizer='adadelta', metrics=["accuracy"])
segnet_basic.compile(loss="sparse_categorical_crossentropy", optimizer='adadelta', metrics=["accuracy"])

# checkpoint
#filepath="weights.best.hdf5"
#checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [tensorboard]



# load the data
# train_data = np.load('./data/train_data.npy')
# train_label = np.load('./data/train_label.npy')
#
# valid_data = np.load('./data/val_data.npy')
# valid_label = np.load('./data/val_label.npy')


# test_data = np.load('./data/test_data.npy')
# test_label = np.load('./data/test_label.npy')

# Getting list of files

DataPath = './CamVid/'
# train_list_fname = DataPath + 'train.txt'
# val_list_fname   = DataPath + 'val.txt'
# img_root         = DataPath + 'train'
# mask_root        = DataPath + 'val'


# def build_abs_paths(basenames):
#         )
#         img_fnames = [os.path.join(os.getcwd() + txt[i][0][7:])  for f in basenames]
#         mask_fnames = [os.path.join(mask_root, f) + '.png' for f in basenames]
#         return img_fnames, mask_fnames
#
# train_basenames = [l.strip() for l in open(train_list_fname).readlines()]
# val_basenames = [l.strip() for l in open(val_list_fname).readlines()][:500]
#
# train_img_fnames, train_mask_fnames = build_abs_paths(train_basenames)
# val_img_fnames, val_mask_fnames = build_abs_paths(val_basenames)


def load_filenames(mode):
    data = []
    label = []
    with open(DataPath + mode +'.txt') as f:
        #print("path found")
        txt = f.readlines()
        txt = [line.split(' ') for line in txt]
    #data.append(np.rollaxis(normalized(cv2.imread(os.getcwd() + txt[i][0][7:])),2))
    #data.append(np.rollaxis(normalized(cv2.imread(os.getcwd() + txt[i][0][7:])),0))
    count = 0
    print("Test print",os.getcwd() +txt[count][0][7:])
    print("Test print sophist",os.getcwd() +txt[count][0][7:])
    print("Label print",os.getcwd() +txt[count][1][7:][:-1])

    data =  [os.getcwd() +txt[count][0][7:] for i in range(len(txt))]
    label =  [os.getcwd() +txt[count][1][7:][:-1] for i in range(len(txt))]
    print('.',end='')
    return np.array(data), np.array(label)


train_img_fnames, train_mask_fnames = load_filenames("train")
val_img_fnames, val_mask_fnames     = load_filenames("val")




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
            img_target_size=(256, 256),
            mask_target_size=(256, 256)),
        samples_per_epoch=len(train_img_fnames),
        nb_epoch=20,
        validation_data=datagen_val.flow_from_list(
            val_img_fnames,
            val_mask_fnames,
            batch_size=8,
            img_target_size=(256, 256),
            mask_target_size=(256, 256)),
        nb_val_samples=len(val_img_fnames),
        callbacks=callbacks_list)





# training_generator = DataGenerator(**params).generate(train_label, train_data)
# print("getting validation generator")
# validation_generator = DataGenerator(**params).generate(valid_label, valid_data)
#
# #validation_generator = DataGenerator(**params).generate(labels, partition['validation'])
# #
# #
# # # Train model on dataset
# history=segnet_basic.fit_generator(generator = training_generator,
#                     steps_per_epoch = len(train_data)//batch_size,
#                     epochs=nb_epoch,
#                     validation_data = validation_generator,
#                     validation_steps = len(valid_data)//batch_size,
#                     class_weight=class_weighting,
#                     callbacks=callbacks_list
#                     )
#


# Fit the model
# history = segnet_basic.fit(train_data, train_label, callbacks=callbacks_list, batch_size=batch_size, epochs=nb_epoch,
#                     verbose=1, class_weight=class_weighting , validation_data=(test_data, test_label), shuffle=True) # validation_split=0.33

# This save the trained model weights to this file with number of epochs
# segnet_basic.save_weights('weights/model_weight_{}.hdf5'.format(nb_epoch))
