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


def get_model_layer_output(model):
    print("Getting model Details")
    inp = model.input                                                  # input placeholder
    outputs = [layer.output for layer in model.layers]          # all layer outputs
    functor = K.function([inp]+ [K.learning_phase()], outputs )        # evaluation function
    print("Getting outputs and functions")

    # Testing
    input_shape=(360, 480, 3)
    test = np.random.random(input_shape)[np.newaxis,...]
    layer_outs = functor([test, 1.])
    print (np.array(layer_outs).shape)




def main():

    with open('segNet_basic_model.json') as model_file:
            segnet_basic = models.model_from_json(model_file.read())
    get_model_layer_output(segnet_basic)

if __name__ == "__main__":
    main()
