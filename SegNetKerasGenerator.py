from __future__ import absolute_import
from __future__ import print_function
import os

from keras import backend as K
from keras import models
from keras.optimizers import SGD, Adadelta
from keras.utils import plot_model
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import (load_img, img_to_array,flip_axis, ImageDataGenerator)

import cv2
import numpy as np
import json

#from DataGenerator import DataGenerator
from utils.image_reader import (
    RandomTransformer,
    SegmentationDataGenerator)

np.random.seed(7) # 0bserver07 for reproducibility
data_shape = 256*256
class_weighting= [0.2595, 0.1826, 4.5640, 0.1417, 0.5051, 0.3826, 9.6446, 1.8418, 6.6823, 6.2478, 3.0, 7.3614]
nb_epoch = 10
batch_size = 3
test_network = False


# load the model:
with open('segNet_basic_model.json') as model_file:
    segnet_basic = models.model_from_json(model_file.read())
print(segnet_basic.summary())


#segnet_basic.compile(loss="categorical_crossentropy", optimizer='adadelta', metrics=["accuracy"])
segnet_basic.compile(loss="sparse_categorical_crossentropy", optimizer='adadelta', metrics=["sparse_categorical_accuracy"])

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


def load_img_array(fname, grayscale=False, target_size=None, dim_ordering='default', data_format=None):
    """Loads and image file and returns an array."""
    img = load_img(fname,
                   grayscale=grayscale,
                   target_size=target_size)
    #x = img_to_array(img, dim_ordering=dim_ordering)
    x = img_to_array(img, data_format=data_format)
    #print("shape of image",x.shape)
    return x


train_img_fnames, train_mask_fnames = load_filenames("train")
val_img_fnames, val_mask_fnames     = load_filenames("val")

#
#
#
images  = [load_img_array(f)for f in train_img_fnames]
masks   = [load_img_array(f, grayscale=True) for f in train_mask_fnames]


print("loading data")

# load the data
# images = np.load('./data/train_data.npy')
# masks = np.load('./data/train_label.npy')
#
#
# valid_data = np.load('./data/val_data.npy')
# valid_label = np.load('./data/val_label.npy')


print("load completed")



#shape of image and mask
print("Shape of input image array is : ",np.array(images).shape)
print("Shape of masks image array is : ",np.array(masks).shape)






# reduce data set for testing
if test_network==True:
    train_img_fnames = train_img_fnames[:6]
    train_mask_fnames = train_mask_fnames[:6]
    val_img_fnames = val_img_fnames[:6]
    val_mask_fnames = val_mask_fnames[:6]


# we create two instances with the same arguments
data_gen_args = dict(featurewise_center=True,
                     featurewise_std_normalization=True,
                     rotation_range=90.,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     zoom_range=0.2)
image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

# Provide the same seed and keyword arguments to the fit and flow methods
seed = 1
image_datagen.fit(images, augment=True, seed=seed)
mask_datagen.fit(masks, augment=True, seed=seed)


image_generator = image_datagen.flow_from_directory(
    directory='CamVid/data/images',
    color_mode='rgb',
    class_mode=None,
    target_size=(192,256),
    batch_size=3,
    seed=seed)



# class mode could be specified if needed (categorical", "binary", "sparse", "input" or None)
# save_to_dir to see augmentation
mask_generator = mask_datagen.flow_from_directory(
    directory='CamVid/data/mask',
    color_mode='grayscale',
    class_mode=None,
    target_size=(192,256),
    batch_size=3,
    seed=seed)

#print("mask_generator generators",mask_generator.classes)
# combine generators into one which yields image and masks
train_generator = zip(image_generator, mask_generator)

segnet_basic.fit_generator(
    train_generator,
    steps_per_epoch=len(train_img_fnames)//batch_size,
    epochs=nb_epoch)










# Generators
#
# transformer_train = RandomTransformer(horizontal_flip=True, vertical_flip=True)
# datagen_train     = SegmentationDataGenerator(transformer_train)
#
# transformer_val   = RandomTransformer(horizontal_flip=False, vertical_flip=False)
# datagen_val       = SegmentationDataGenerator(transformer_val)
#
#
#
# #print()
# segnet_basic.fit_generator(
#         datagen_train.flow_from_list(
#             train_img_fnames,
#             train_mask_fnames,
#             shuffle=True,
#             batch_size=batch_size,
#             img_target_size=(256, 256),
#             mask_target_size=(256, 256)),
#             samples_per_epoch=len(train_img_fnames)//batch_size,
#             nb_epoch=nb_epoch,
#             validation_data=datagen_val.flow_from_list(
#                 val_img_fnames,
#                 val_mask_fnames,
#                 batch_size=batch_size,
#                 img_target_size=(256, 256),
#                 mask_target_size=(256, 256)),
#                 nb_val_samples=len(val_img_fnames),
#                 callbacks=callbacks_list)

# This save the trained model weights to this file with number of epochs
segnet_basic.save_weights('weights/model_weight_{}.hdf5'.format(nb_epoch))
