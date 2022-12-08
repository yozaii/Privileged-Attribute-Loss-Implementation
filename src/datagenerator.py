# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 13:30:09 2022

@author: Youssef
"""

import tensorflow as tf
import keras
import numpy as np
import skimage.io as sk
from data_utils import *

# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, filepaths, batch_size=16, im_dim=(224,224,3), h_dim = (112,112,1),
                 n_classes=7, shuffle=True):
        'Initialization'
        self.filepaths = filepaths
        self.batch_size = batch_size
        self.im_dim = im_dim
        self.h_dim = h_dim
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_filepaths_temp = [self.filepaths[k] for k in indexes]

        # Generate data
        x, h, y = self.__data_generation(list_filepaths_temp)

        return x, h, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.filepaths))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_filepaths_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        x = np.empty((self.batch_size, *self.im_dim))
        h = np.empty((self.batch_size, *self.h_dim))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, filepath in enumerate(list_filepaths_temp):
            
            # image
            x[i] = load_img(filepath[0])
            
            # heatmap
            h[i] = load_heatmap(filepath[1])

            # class label
            y[i] = int(filepath[2])

        return x, h, keras.utils.to_categorical(y, num_classes=self.n_classes)
    
    
if __name__ == '__main__':
    
    pass