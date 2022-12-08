import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Layer, Conv2D, MaxPool2D, Flatten, Dense, Input
from train import *


class PALModel(keras.Model):
    
    def __init__(self, weights_path = None,  num_classes = 7, backbone_name = 'vgg16', n_PAL_layer = 13 ):
        
        super().__init__()
        self.num_classes = num_classes
        self.n_PAL_layer = n_PAL_layer
        self.weights_path = weights_path
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
        
        print("x:", x)
        print("h:", h)
        print("y:", y)
        
        print(x[0])
        
        # # Separate x into two : x (images) and heatmaps
        # x, heatmaps = x
        

        with tf.GradientTape() as tape:
            
            # Forward pass and output vector sum
            y_pred = self(x, training=True)
            # f_o = keras.math.abs(self.layers[-1].output)
            # f_o_sum = keras.math.reduce_sum(f_o)
            
            # B loss (categorical cross entropy in our case)
            # loss_pred = self.compiled_loss(y, y_pred)
        
        # Initialize PAL attribution
        PAL_layer = self.layers[self.n_PAL_layer].output
        
        # number of channels to use for PAL
        channels = PAL_layer.shape[3]
    
        # Compute gradients of attribution
        attribution = x*tape.gradient(f_o_sum, self.trainable_variables)
        
        # Calculate Privilegd Attribution Loss and add it to total loss
        PAL_loss = calc_PAL(attribution, heatmaps, channels) 
        loss_total = loss_pred + PAL_loss
        
        # Back prop
        gradients = tape.gradient(loss_total, self.trainable_variables)
        
        # Update weights
        self.opt.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}
    



    def compile(self, loss ='categorical_crossentropy', optimizer = 'adam', run_eagerly = True):
        
        super().compile()
        self.opt = optimizer
        self.l = loss