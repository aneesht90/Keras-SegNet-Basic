from __future__ import absolute_import
from __future__ import print_function
import os


import keras.models as models
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape, Permute
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization


from keras import backend as K

import cv2
import numpy as np
import json
np.random.seed(7) # 0bserver07 for reproducibility



segment_count = 12
channel_count = 3
image_height  = 360
image_width   = 480

filter_size_enc_1 = 64
filter_size_enc_2 = 128
filter_size_enc_3 = 256
filter_size_enc_4 = 512

filter_size_dec_1 = 512
filter_size_dec_2 = 256
filter_size_dec_3 = 128
filter_size_dec_4 = 64


input_shape   = (image_height, image_width, channel_count)
output_shape  = (image_height, image_width, segment_count)
data_shape    = image_height * image_width
data_shape_   = (image_height, image_width)




def create_encoding_layers():
    kernel = 3
    #filter_size = 64
    pad = 1
    pool_size = 2
    return [
        #ZeroPadding2D(padding=(pad,pad)),
        #Conv2D(filter_size, (kernel, kernel), padding='valid'),
        Conv2D(filter_size_enc_1, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(pool_size, pool_size)),

        #ZeroPadding2D(padding=(pad,pad)),
        #Conv2D(128, (kernel, kernel), padding='valid'),
        Conv2D(filter_size_enc_2, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(pool_size, pool_size)),

        #ZeroPadding2D(padding=(pad,pad)),
        #Conv2D(256, (kernel, kernel), padding='valid'),
        Conv2D(filter_size_enc_3, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(pool_size, pool_size)),
        Dropout(0.5),

        #ZeroPadding2D(padding=(pad,pad)),
        #Conv2D(512, (kernel, kernel), padding='valid'),
        Conv2D(filter_size_enc_4, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
    ]

def create_decoding_layers():
    kernel = 3
    #filter_size = 64
    pad = 1
    pool_size = 2
    return[
        # ZeroPadding2D(padding=(pad,pad)),
        # Conv2D(512, (kernel, kernel), padding='valid'),
        Conv2D(filter_size_dec_1, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Dropout(0.5),

        UpSampling2D(size=(pool_size,pool_size)),
        # ZeroPadding2D(padding=(pad,pad)),
        # Conv2D(256, (kernel, kernel), padding='valid'),
        Conv2D(filter_size_dec_2, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Dropout(0.5),

        UpSampling2D(size=(pool_size,pool_size)),
        # ZeroPadding2D(padding=(pad,pad)),
        # Conv2D(128, (kernel, kernel), padding='valid'),
        Conv2D(filter_size_dec_3, (kernel, kernel), padding='same'),
        BatchNormalization(),

        UpSampling2D(size=(pool_size,pool_size)),
        # ZeroPadding2D(padding=(pad,pad)),
        # Conv2D(filter_size, (kernel, kernel), padding='valid'),
        Conv2D(filter_size_dec_4, (kernel, kernel), padding='same'),
        BatchNormalization(),
    ]




segnet_basic = models.Sequential()

segnet_basic.add(Layer(input_shape=input_shape))



segnet_basic.encoding_layers = create_encoding_layers()
for l in segnet_basic.encoding_layers:
    segnet_basic.add(l)

# Note: it this looks weird, that is because of adding Each Layer using that for loop
# instead of re-writting mode.add(somelayer+params) everytime.

segnet_basic.decoding_layers = create_decoding_layers()
for l in segnet_basic.decoding_layers:
    segnet_basic.add(l)


segnet_basic.add(Conv2D(segment_count, (1, 1), padding='valid',))

#segnet_basic.add(Reshape((segment_count,data_shape), input_shape=output_shape))
segnet_basic.add(Reshape((image_height,image_width,segment_count), input_shape=output_shape))
#segnet_basic.add(Permute((2, 1)))
segnet_basic.add(Activation('softmax'))



# Save model to JSON

with open('segNet_basic_dropout_model.json', 'w') as outfile:
    outfile.write(json.dumps(json.loads(segnet_basic.to_json()), indent=2))
