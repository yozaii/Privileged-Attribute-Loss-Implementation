# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 17:09:15 2022

@author: Youssef
"""

from model import *

dataset_x = tf.data.Dataset.from_tensor_slices((train_x, train_h))
dataset_y = tf.data.Dataset.from_tensor_slices(train_y)
train_data = tf.data.Dataset.zip((dataset_x, dataset_y)).batch(16).repeat()

sys.path.append('../')
VGG16_WEIGHTS_PATH = '../pretrained_models/rcmalli_vggface_tf_notop_vgg16.h5'

# ============================================================= #

# filename = '../data/RAFDB/raw/Image/aligned/test_0001_aligned.jpg'
# filename2 = '../data/RAFDB/preprocessed/landmarks/'
# im = sk.imread(filename)

# xx = keras.applications.VGG16(include_top=(True))
# print(len(xx.layers))
# xx.summary()

# ============================================================= #

m = PALModel(weights_path = VGG16_WEIGHTS_PATH)     

opt = keras.optimizers.Adam(learning_rate = 0.0005)
loss_v = keras.losses.CategoricalCrossentropy()
m.compile(loss = loss_v, optimizer=opt)

m.fit(train_data, epochs = 1, steps_per_epoch=4)
m.summary()