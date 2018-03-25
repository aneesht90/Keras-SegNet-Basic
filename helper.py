from __future__ import absolute_import
from __future__ import print_function

import cv2
import numpy as np
import itertools

from helper import *
import os



image_height  = 192
image_width   = 256
segment_count = 12


def normalized(rgb):
    #return rgb/255.0
    norm=np.zeros((rgb.shape[0], rgb.shape[1], 3),np.float32)

    b=rgb[:,:,0]
    g=rgb[:,:,1]
    r=rgb[:,:,2]

    norm[:,:,0]=cv2.equalizeHist(b)
    norm[:,:,1]=cv2.equalizeHist(g)
    norm[:,:,2]=cv2.equalizeHist(r)

    return norm

def one_hot_it(labels):
    x = np.zeros([image_height,image_width,segment_count])
    for i in range(image_height):
        for j in range(image_width):
            # if labels[i][j]>=12:
            #     labels[i][j]=11
                #print("index is 12")
            x[i,j,labels[i][j]]=1
    return x
