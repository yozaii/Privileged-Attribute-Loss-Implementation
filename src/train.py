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
    
def custom_train(model, dataset, epochs, batch_size = 16):

    train_size = len(dataset[0][0])
    print('we here')

    for epoch in range(epochs):

        # Iterate over the batches of the dataset.
        for step in range(0,train_size,batch_size):
            
            # load a batch and get images / heatmaps
            x, y = load_batch(dataset, step, step+batch_size -1)
            x, heatmaps = x

            # Open a GradientTape to record the operations run
            # during the forward pass, which enables auto-differentiation.
            with tf.GradientTape() as tape:
    
               # Forward pass and output vector sum
               y_pred = model(x, training=True)

               # Initialize PAL attribution
               PAL_layer = model.layers[model.n_PAL_layer].output
               attribution_loss = 0
               
               # number of channels
               channels = PAL_layer.shape[3]
               
               # width and height of attribution
               width = PAL_layer.shape[1]
               height = PAL_layer.shape[2]
               
               # Calculate PAL (attribution loss)
               for h in heatmaps:
                   # resize heatmap with respect to attribution (its width and height)
                   h = resize(h, (width, height))
                   # Get PAL loss of n_PAL_layer (attribution map of nth layer)
                   attribution_loss += PAL_loss(PAL_layer, h, channels)

               # Output loss (categorical cross entropy in our case)
               loss = model.compiled_loss(y, y_pred)
               loss_total = attribution_loss + loss
       
            # Compute gradients
            trainable_vars = model.trainable_variables
            gradients = x*tape.gradient(loss_total, trainable_vars)
            # Update weights
            model.opt.apply_gradients(zip(gradients, trainable_vars))
            # Update metrics (includes the metric that tracks the loss)
            model.compiled_metrics.update_state(y, y_pred)
            # # Return a dict mapping metric names to current value
            # return {m.name: m.result() for m in self.metrics}
             
            # Log every 200 batches.
            if step % 200 == 0:
                print(
                    "Training loss (for one batch) at step %d: %.4f"
                    % (step, float(loss_value))
                )
                print("Seen so far: %s samples" % ((step + 1) * batch_size))