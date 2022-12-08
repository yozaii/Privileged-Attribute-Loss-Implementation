# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 17:09:15 2022

@author: Youssef
"""

from model import *


# tf.config.run_functions_eagerly(True)

# print(tf.executing_eagerly())

# dataset_x = tf.data.Dataset.from_tensor_slices((train_x, train_h))
# dataset_y = tf.data.Dataset.from_tensor_slices(train_y)
# train_data = tf.data.Dataset.zip((dataset_x, dataset_y)).batch(16)



# ============================================================= #

# filename = '../data/RAFDB/raw/Image/aligned/test_0001_aligned.jpg'
# filename2 = '../data/RAFDB/preprocessed/landmarks/'
# im = sk.imread(filename)

# xx = keras.applications.VGG16(include_top=(True))
# print(len(xx.layers))
# xx.summary()

# ============================================================= #

# tf.config.run_functions_eagerly(True)
sys.path.append('../')
sys.path.append('../data/RAFDB/raw/Image/aligned/')
VGG16_WEIGHTS_PATH = '../pretrained_models/rcmalli_vggface_tf_notop_vgg16.h5'

im_dir = '../data/RAFDB/raw/Image/aligned/'
h_dir = '../data/RAFDB/raw/landmarks/'

train_dataset_partition, test_dataset_partition = load_partition()
training_generator = DataGenerator(train_dataset_partition)

m = PALModel(weights_path = VGG16_WEIGHTS_PATH)    

opt = keras.optimizers.Adam(learning_rate = 0.0005)
loss_v = keras.losses.CategoricalCrossentropy()
m.compile(loss = loss_v, optimizer=opt)

# train
m.fit(training_generator, epochs = 2,  steps_per_epoch=2)

# path to save model, and then save it
save_dir = '../pretrained_models/model_sanity_check'
m.save(save_dir)