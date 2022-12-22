# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 17:09:15 2022

@author: Youssef
"""

from model import *
from keras.metrics import Accuracy, CategoricalAccuracy, CategoricalCrossentropy


if __name__ == '__main__':
    
    # Add root directory to path
    sys.path.append('../')
    
    # Load model weights
    VGG16_WEIGHTS_PATH = '../pretrained_models/rcmalli_vggface_tf_notop_vgg16.h5'
    
    # Dataset directories (images / heatmaps)
    im_dir = '../data/RAFDB/raw/Image/aligned/'
    h_dir = '../data/RAFDB/raw/landmarks/'
    
    # Load dataset filepaths and pass them to the generator
    train_dataset_partition, test_dataset_partition = load_partition()
    training_generator = DataGenerator(train_dataset_partition)
    
    # Create model
    m = PALModel(weights_path = VGG16_WEIGHTS_PATH)     
    
    # Model compilation params
    opt = keras.optimizers.Adam(learning_rate = 0.0005)
    loss_fn = keras.losses.CategoricalCrossentropy()
    met = keras.metrics.CategoricalAccuracy()
    
    # compile model
    m.compile(loss = loss_fn, optimizer=opt, metrics = met)
    
    # train
    m.fit(training_generator, epochs = 2)
    
    # path to save model, and then save it
    save_dir = '../pretrained_models/model_sanity_check'