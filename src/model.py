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

        # Unpack the data
        x, y = data
        

        # heatmap = heatmap(x)
    

        x, heatmaps = x
        

        with tf.GradientTape() as tape:
            
            y_pred = self(x, training=True)  # Forward pass
        

            
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            
            # Get PAL loss of n_PAL_layer (attribution map of nth layer)
            PAL_loss(self.layers[self.n_PAL_layer], prior_heatmap, channels)

            # channel strategy
            channels = x.shape[3]
            
            PAL_loss = 0
            for heatmap in heatmaps:
            
                # Get PAL loss of n_PAL_layer (attribution map of nth layer)
                PAL_loss += PAL_loss(self.layers[self.n_PAL_layer], heatmap, channels)

            
            # Output loss (categorical cross entropy in our case)
            loss = self.compiled_loss(y, y_pred)
            loss_total = PAL_loss + loss
    
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = x*tape.gradient(loss, trainable_vars)
        # Update weights
        self.opt.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}
    

    def compile(self, loss ='categorical_crossentropy', optimizer = 'Adam'):
        
        self.optimizer = optimizer
        self.loss = loss

        pass



    def compile(self, loss ='categorical_crossentropy', optimizer = 'adam'):
        
        super().compile()
        self.opt = optimizer
        self.l = loss
        print('TEST')
        
    def custom_train(batch_size, epochs):
        
        self.fit()
