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




def post_processing(image_dir):
    pass

if __name__ == "__main__":
    post_processing('CamVid/train/output')
