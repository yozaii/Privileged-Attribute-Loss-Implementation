import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Layer, Conv2D, MaxPool2D, Flatten, Dense
from data_utils import *


def calc_PAL(attribution, prior_heatmaps, channels):
    """
    Privileged attribution loss
    Parameters
    ----------
    attribution : TYPE
        DESCRIPTION. attribution of layer with respect to CNN output
    prior_heatmap : TYPE
        DESCRIPTION. image of heatmap of facial landmarks
    channels : TYPE
        DESCRIPTION. number of channels to be used according to channel strategy

    Returns
    -------
    None.

    """
    # batch size
    batch_size = prior_heatmaps.shape[0]
    
    # width and height of attribution
    width = PAL_layer.shape[1]
    height = PAL_layer.shape[2]  
    
    # resize prior_heatamp to match size of attribution layer
    prior_heatmaps = tf.image.resize(prior_heatmaps, attribution[0], attribution[1])
    
    # Total privileged attribution loss
    total_PAL = 0    
    
    
    for i in range(prior_heatmaps.shape[0]):
        
        # resize prior_heatamp to match size of attribution layer
        prior_heatmap = tf.image.resize(prior_heatmap, attribution[0], attribution[1])
        
        
        for c in range(channels):
            
            att_c = attribution[:,:,:,c] # attribution of one channel
            
            # cross correlation parameters
            mu = np.sum(att_c)
            sigma_sq = np.sum((att_c - mu) *(att_c-mu))
            sigma = np.sqrt(sigma_sq)
            
            # Priveleged attribution loss formula for a channel c
            conv = np.convolve((att_c - mu)/sigma, prior_heatmap)
            pal = -np.sum(conv)
            total_PAL += pal
            
        return total_PAL
    
    
class CustomCallback(keras.callbacks.Callback):
    

    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())
        print("End epoch {} of training; got log keys: {}".format(epoch, keys))