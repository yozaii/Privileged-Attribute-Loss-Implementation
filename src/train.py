import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Layer, Conv2D, MaxPool2D, Flatten, Dense
from data_utils import *


def PAL_loss(attribution, prior_heatmap, channels):
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
    
    # resize prior_heatamp to match size of attribution layer
    prior_heatmap = img_resize(prior_heatmap, attribution[0], attribution[1])
    
    # Total privileged attribution loss
    total_PAL = 0
    
    for c in range(channels):
        
        att_c = attribution[:,:,c] # attributin of one channel
        
        # cross correlation parameters
        mu = np.sum(att_c)
        sigma_sq = np.sum((att_c - mu) *(att_c-mu))
        sigma = np.sqrt(sigma_sq)
        
        # Priveleged attribution loss formula for a channel c
        conv = np.convolve((att_c - mu)/sigma, prior_heatmap)
        pal = -np.sum(conv)
        total_PAL += pal
        
    return total_PAL
    