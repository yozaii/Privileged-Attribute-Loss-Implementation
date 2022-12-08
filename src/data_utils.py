import numpy as np
import tensorflow as tf
from tensorflow import keras

import skimage.io as sk
from skimage.filters import gaussian
from skimage.util import img_as_float
from skimage.transform import rescale, resize

import sys
import os
import time

sys.path.append('../')

# ================ GLOBAL VARS ================= #

# Directories

ALIGNED_IMAGE_DIRPATH = ('../data/RAFDB/raw/Image/aligned/')
LANDMARKS_DIRPATH = ('../data/RAFDB/raw/landmarks/')

# Files

PARTITION_FILEPATH = ('../data/RAFDB/raw/EmoLabel/list_patition_label.txt')

# ======================================================= #
# ================ DATA WRITING FUNCTIONS =============== #
# ======================================================= #


# https://www.tensorflow.org/tutorials/load_data/images


def download_dataset_to_dir():
    pass

def save_model_to_dir():
    pass
    

# ======================================================= #
# ================ DATA LOADING UTILS =================== #
# ======================================================= #

def load_partition(filepath = None):


    # To use global variable if no filepath is given
    if filepath == None:
        filepath = PARTITION_FILEPATH
        
    # Open file and read all lines
    f = open(filepath, 'r')
    Lines = f.readlines()
    
    # list for filenames (ex : test_aligned_0001), and its label (0-6)
    train_filenames = list()
    test_filenames = list()
    train_label = {}
    test_label = {}
    for line in Lines:
        
        # x and y coordinates of landmarks
        x, y = line.split()
        
        # Remove filename extension
        x = x.replace('.jpg','')
        
        # testing second letter of string by checking if t(r)ain not t(e)st
        if line[1] == 'r':
            train_label[x] = float(y)
            train_filenames.append(x)
            
        else :
            test_label[x] = float(y)
            test_filenames.append(x)
        
    f.close()
        
    return (train_filenames, train_label), (test_filenames, test_label)

def load_heatmap(filepath, im_h = 112, im_w = 112, sigma = 3):

    # Open file and read all lines
    f = open(filepath, 'r')
    Lines = f.readlines()
    
    # List of heatmaps for each landmark
    heatmap = np.zeros((im_h,im_w, 1), dtype = np.float32)
    
    for line in Lines:
        
        # x and y coordinates of landmarks
        x, y = line.split()
        x = int(x)
        y = int(y)
        
        # Landmark coordinates is set to 1, everything else is 0
        heatmap[x,y] = 1
        
        
        
    f.close()

    heatmap = gaussian(heatmap, sigma)
    
    return heatmap

def load_img(filepath):
    
    img = sk.imread(filepath)
    img = img.astype(np.float32)/255
    return img

# ======================= CLASSES ======================== #

# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, partition, batch_size=16, im_dim=(224,224,3), h_dim = (112,112,1),
                 n_classes=7, shuffle=True):
        'Initialization'
        self.filepaths = partition[0]
        self.label = partition[1]
        self.batch_size = batch_size
        self.im_dim = im_dim
        self.h_dim = h_dim
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.filepaths) / self.batch_size))

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
            x[i] = load_img(ALIGNED_IMAGE_DIRPATH + filepath + '_aligned.jpg')
            
            # heatmap
            h[i] = load_heatmap(LANDMARKS_DIRPATH + filepath + '_aligned.txt')

            # class label
            y[i] = self.label[filepath] - 1

        return x, h, keras.utils.to_categorical(y, num_classes=self.n_classes)


if __name__ == '__main__':
    
    
    sys.path.append('../')
    

    filename = '../data/RAFDB/raw/landmarks/test_0006_aligned.txt'
    dirname = '../data/RAFDB/raw/landmarks'
    
    
    # ====================================================== #
   # # tf.config.run_functions_eagerly(True)
   # sys.path.append('../')
   # VGG16_WEIGHTS_PATH = '../pretrained_models/rcmalli_vggface_tf_notop_vgg16.h5'

   # im_dir = '../data/RAFDB/raw/Image/aligned/'
   # h_dir = '../data/RAFDB/raw/landmarks/'

   # dataset_filepaths = load_dataset_filepaths(im_dir, h_dir)

   # x, h , y = dataset_filepaths
   # x_train, x_test = x
   # h_train, h_test = h
   # y_train, y_test = y

   # # Load training dataset
   # train_dataset = tf.data.Dataset.from_tensor_slices((x_train, h_train, h_train))

    # ====================================================== #
    
    # # start time
    # start = time.time()
    
    # data, y = load_partition()
    # train_y, test_y = y
    # train_y = keras.utils.to_categorical(train_y)
    # test_y = keras.utils.to_categorical(test_y)
    
    
    
    # img_filenames = list()
    # heatmap_filenames = list()
    
    # for dataname in data:
    #     img_filenames.append(dataname + '_aligned.jpg')
    #     heatmap_filenames.append(dataname + '_aligned.txt')
        
    # print('after filenames', time.time() - start)
    
    # img_dir = '../data/RAFDB/raw/Image/aligned/'
    # heatmap_dir = '../data/RAFDB/raw/landmarks/'
      

    # train_h, test_h = load_all_heatmaps(heatmap_dir, heatmap_filenames, 112, 112)
    # print(time.time() - start)
    
    # train_x, test_x = load_all_imgs(img_dir, img_filenames)
    
    # print(time.time() - start)
    # train_x = train_x.astype(np.float32)/255
    # test_x = train_x.astype(np.float32)/255
    
    
    
    
    # # end time
    # end = time.time()
    
    # elapsed = end - start
    # print(elapsed)
    
    # ====================================================== #
    
    # filename = '../data/RAFDB/raw/Image/aligned/test_0006_aligned.jpg'
    # im = sk.imread(filename)
    # im = img_as_float(im)
    
    im_dir = '../data/RAFDB/raw/Image/aligned/'
    h_dir = '../data/RAFDB/raw/landmarks/'
    
    dataset_filepaths = load_dataset_filepaths(im_dir, h_dir)
    
    x, y, h = dataset_filepaths
    print(x[0][1])

    