import json
import sys
import os
import numpy as np
import pandas as pd
from skimage.io import imread
from matplotlib import pyplot as plt

from keras import backend as K
from keras import models
from keras.optimizers import SGD
from keras.utils import plot_model


def get_model_memory_usage(batch_size, model):
    import numpy as np
    from keras import backend as K

    shapes_mem_count = 0
    for l in model.layers:
        single_layer_mem = 1
        for s in l.output_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
    non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])

    total_memory = 4.0*batch_size*(shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3)
    return gbytes


def main():

    with open('segNet_basic_model.json') as model_file:
            segnet_basic = models.model_from_json(model_file.read())
    print(get_model_memory_usage(1  , segnet_basic))

if __name__ == "__main__":
    main()
