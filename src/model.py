import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Layer, Conv2D, MaxPool2D, Flatten, Dense, Input
from train import *


class PALModel(keras.Model):
    
    def __init__(self, weights_path = None,  num_classes = 7, backbone_name = 'vgg16', n_PAL_layer = 13,
                 channel_strat = 0):
        
        super().__init__()
        self.num_classes = num_classes
        self.n_PAL_layer = n_PAL_layer
        self.weights_path = weights_path
        self.channel_strat = channel_strat # 0 : all channel, 1 : mean, 2 : half of mean
        self.im_shape = (224,224,3)

        
        # Backbone CNN model choice
        if backbone_name == 'resnet50':
        
            self.backbone = [layer for layer in keras.applications.ResNet50(weights=self.weights_path
                        ,include_top = False, input_shape = self.im_shape).layers]
        
        # VGG16 as default CNN backbone otherwise
        else:
            
            self.backbone = [layer for layer in keras.applications.VGG16(weights=self.weights_path,
                            include_top = False, input_shape = self.im_shape).layers]
                
        # Top part of model (classifier)
        self.flat = Flatten()
        self.dense1 = Dense(units=4096,activation="relu")
        self.dense2 = Dense(units=4096,activation="relu")
        self.classifier = Dense(units=7, activation="softmax")
        self.build((None, 224, 224, 3))


    def call(self, inputs):
        
        x = inputs
        # We iterate over all layers to have a more visually appealing model.summary()
        for layer in self.backbone:
            x = layer(x)
            
        # Chain the top of the model with the backbone
        x = self.flat(x)
        x = self.dense1(x)
        x = self. dense2(x)
        x = self.classifier(x)
        
        return x

    # https://keras.io/guides/customizing_what_happens_in_fit/
    def train_step(self, data):

        # print(data)
        
        # Unpack the data
        x, h, y = data
        
        
        
        with tf.GradientTape() as tape:
            
            # Forward pass
            y_pred = self(x, training=True)
            
            # output vector sum
            f_o_sum = tf.reduce_sum(y_pred, axis = 1)

            
            # Backprop loss (categorical cross entropy in our case)
            loss_pred = self.compiled_loss(y, y_pred)
        
        
        # Initialize attribution
        attribution_layer = self.layers[self.n_PAL_layer].output
    
        # # Compute gradients of attribution
        # attribution = x*tape.gradient(f_o_sum, attribution_layer)
        
        # # Choose channel strategy:
        # if self.channel_strat == 0:
        #     num_channels = attribution.shape[3]
        # elif self.channel_strat == 1:
        #     num_channels = attribution.shape[3]
        #     attribution = (1/num_channels)
        # elif self.channel_strat == 2:
        #     num_channels = int(attribution_layer.shape[3]/2)
        #     attribution = tf.slice(attribution, )
        
        # # Calculate Privilegd Attribution Loss and add it to total loss
        # PAL_loss = calc_PAL(attribution, heatmaps, num_channels)
        
        
        loss_total = loss_pred # + PAL_loss
        
        # Back prop
        gradients = tape.gradient(loss_total, self.trainable_variables)
        
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        # Update metrics (includes the metric that tracks the loss)
     
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        
        
        return {m.name: m.result() for m in self.metrics}