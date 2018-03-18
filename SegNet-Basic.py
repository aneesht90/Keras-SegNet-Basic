from __future__ import absolute_import
from __future__ import print_function
import os

# os.environ['KERAS_BACKEND'] = 'theano'
# os.environ['THEANO_FLAGS']='mode=FAST_RUN,device=gpu0,floatX=float32,optimizer=None'
#
# import keras.models as models
# from keras.layers.core import Layer, Dense, Dropout, Flatten, Reshape, Permute
# from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
# from keras.layers.normalization import BatchNormalization
# from keras.callbacks import ModelCheckpoint
#
# from keras import backend as K



from keras import backend as K
from keras import models
from keras.optimizers import SGD, Adadelta
from keras.utils import plot_model
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint


import cv2
import numpy as np
import json
np.random.seed(7) # 0bserver07 for reproducibility

data_shape = 360*480

class_weighting= [0.2595, 0.1826, 4.5640, 0.1417, 0.5051, 0.3826, 9.6446, 1.8418, 6.6823, 6.2478, 3.0, 7.3614]

nb_epoch = 1
batch_size = 3



tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0,
                              write_graph=True, write_images=False)


# load the model:
with open('segNet_basic_model.json') as model_file:
    segnet_basic = models.model_from_json(model_file.read())


segnet_basic.compile(loss="categorical_crossentropy", optimizer='adadelta', metrics=["accuracy"])

# checkpoint
filepath="weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
#callbacks_list = [checkpoint]
callbacks_list = [tensorboard]



# load the data
train_data = np.load('./data/train_data.npy')
train_label = np.load('./data/train_label.npy')


valid_data = np.load('./data/val_data.npy')
valid_label = np.load('./data/val_label.npy')


# Fit the model
history = segnet_basic.fit(train_data, train_label, batch_size=batch_size, epochs=nb_epoch,
                    verbose=1, class_weight=class_weighting , validation_data=(valid_data, valid_label),
                    shuffle=True,callbacks=callbacks_list) # validation_split=0.33






# This save the trained model weights to this file with number of epochs
segnet_basic.save_weights('weights/model_weight_{}.hdf5'.format(nb_epoch))
