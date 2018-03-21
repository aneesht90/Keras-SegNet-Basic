from __future__ import absolute_import
from __future__ import print_function
import os
import cv2
import numpy as np
import json

from keras import backend as K
from keras import models
from keras.optimizers import SGD, Adadelta
from keras.utils import plot_model
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
np.random.seed(7) # 0bserver07 for reproducibility

image_height  = 360
image_width   = 480
class_weighting= [0.2595, 0.1826, 4.5640, 0.1417, 0.5051, 0.3826, 9.6446, 1.8418, 6.6823, 6.2478, 3.0, 7.3614]
nb_epoch = 10
batch_size = 3

#data_shape    = image_height * image_width



# load the model:
with open('segNet_full_model.json') as model_file:
    segnet_basic = models.model_from_json(model_file.read())
print(segnet_basic.summary())

# segnet_basic.compile(loss="categorical_crossentropy", optimizer='adadelta', metrics=["accuracy"])
#
# # checkpoint
# filepath="weights.best.hdf5"
# checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
# tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False)
# callbacks_list = [checkpoint,tensorboard]
#
# print("loading data")
#
# # load the data
# train_data = np.load('./data/train_data.npy')
# train_label = np.load('./data/train_label.npy')
#
#
# valid_data = np.load('./data/val_data.npy')
# valid_label = np.load('./data/val_label.npy')
#
#
# print("load completed")
#
# # Fit the model
# history = segnet_basic.fit(train_data, train_label, batch_size=batch_size, epochs=nb_epoch,
#                     verbose=1, class_weight=class_weighting , validation_data=(valid_data, valid_label),
#                     shuffle=True, callbacks=callbacks_list) # validation_split=0.33
#
# # This save the trained model weights to this file with number of epochs
# segnet_basic.save_weights('weights/model_weight_{}.hdf5'.format(nb_epoch))
