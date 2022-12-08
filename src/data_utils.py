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

ALIGNED_IMAGE_DIRPATH = ('../data/RAFDB/raw/Image/aligned')
LANDMARKS_DIRPATH = ('../data/RAFDB/raw/landmarks')

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
# =============== DATA LOADING FUNCTIONS ================ #
# ======================================================= #

def load_dataset_filepaths(im_dir, h_dir):
    """

    Parameters
    ----------
    im_dir : TYPE
        DESCRIPTION.
    h_dir : TYPE
        DESCRIPTION.

    Returns
    -------
    train_im_filepaths : TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.
    tuple
        DESCRIPTION.

    """
    
    # 
    filenames, y = load_partition()
    
    img_train_filepaths = list()
    img_test_filepaths = list()
    heatmap_train_filepaths = list()
    
    # from load_partition we get list of y labels, and filenames of x
    filenames_x, y = load_partition()
    y_train, y_test = y
    
    # image filepaths
    train_im_filepaths = list()
    test_im_filepaths = list()
    
    # heatmap filepaths
    train_h_filepaths = list()
    test_h_filepaths = list()
    
    for filename in filenames:
        
        # testing second letter of string by checking if t(r)ain not t(e)st
        if filename[1] == 'r':
            
            train_im_filepaths.append(im_dir + filename + '_aligned.jpg')
            train_h_filepaths.append(h_dir + filename + '_aligned.txt')
            
        else:
            test_im_filepaths.append(im_dir + filename + '_aligned.jpg')
            test_h_filepaths.append(h_dir + filename + '_aligned.txt')
            
    return (train_im_filepaths, test_im_filepaths), (train_h_filepaths, test_h_filepaths),(y_train, y_test)
    
def load_keras_dataset_filepaths(im_dir, h_dir):

    
    dataset_filepaths = load_dataset_filepaths(im_dir, h_dir)

    x, h , y = dataset_filepaths
    x_train, x_test = x
    h_train, h_test = h
    y_train, y_test = y

    # Load training dataset
    dataset_train = tf.data.Dataset.from_tensor_slices((x_train, h_train, h_train))
    dataset_test = tf.data.Dataset.from_tensor_slices((x_test,h_test,y_test))
    
    return dataset_train, dataset_test

def load_partition(filepath = None):


    # To use global variable if no filepath is given
    if filepath == None:
        filepath = PARTITION_FILEPATH
        
    # Open file and read all lines
    f = open(filepath, 'r')
    Lines = f.readlines()
    
    # list for filenames (ex : test_aligned_0001), and its label (0-6)
    filenames = list()
    train_label = list()
    test_label = list()
    for line in Lines:
        
        # x and y coordinates of landmarks
        x, y = line.split()
        
        # testing second letter of string by checking if t(r)ain not t(e)st
        if line[1] == 'r':
            train_label.append(float(y))
            
        else :
            test_label.append(y)
        
        # Remove file extension and append to list
        x = x.replace('.jpg','')
        
        filenames.append(x)
        
    f.close()
        
    return filenames, (train_label, test_label)

def load_heatmap(filepath, im_h = 112, im_w = 112, sigma = 3):

    # Open file and read all lines
    f = open(filepath, 'r')
    Lines = f.readlines()
    
    # List of heatmaps for each landmark
    heatmap = np.zeros((im_h,im_w), dtype = np.float32)
    
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

def load_all_heatmaps(dirname, filenames, im_h, im_w, sigma = 3):
    
    # list that will hold trainset / testset heatmaps
    train_hm = list()
    test_hm = list()
    
    for filename in filenames:

        f = dirname + filename
        heatmap = load_heatmap(f, im_h, im_w, sigma)
        
        # filename's second letter is r -> train heatmap
        if filename[1] == 'r':
            train_hm.append(heatmap)
            
        else :
            test_hm.append(heatmap)
    
    return np.array(train_hm), np.array(test_hm)

def load_img(filepath):
    
    img = sk.imread(filepath)
    img = img.astype(np.float32)/255
    return img

def load_all_imgs(dirname, filenames):
    
    # list that will hold trainset / testset heatmaps
    train_img = list()
    test_img = list()
    
    for filename in filenames:

        f = dirname + filename
        img = load_img(f)
        
        # filename's second letter is r -> train img
        if filename[1] == 'r':
            train_img.append(img)
            
        else :
            test_img.append(img)
    
    return np.array(train_img), np.array(test_img)

def load_data():
    pass
    # return (train_imgs, train_labels), (test_imgs, test_labels)



# ======================= BATCH ========================= #


def load_batch(dataset_filepaths, ind_start, ind_end, train = True):
    
    # Unpack filepaths of images, heatmaps, and labels
    imgs, heatmaps, y = dataset_filepaths
    
    # Initialize batch lists
    imgs_batch = list()
    h_batch = list()
    
    # ind_of_set = 0 if we want to load a batch from training set, 1 otherwise
    if train:
        ind_of_set = 0
    else:
        ind_of_set = 1

    # Create sublists of batches
    y_batch = y[ind_of_set][ind_start:ind_end]
    imgs, heatmaps = imgs[ind_of_set][ind_start:ind_end], heatmaps[ind_of_set][ind_start:ind_end]
    
    # Iterate over batches to create images / heatmaps from their filepaths
    for (im, h) in zip(imgs,heatmaps):
        imgs_batch.append(load_img(im))
        h_batch.append(load_heatmap(h))
        
    # Change lists to ndarray
    imgs_batch = np.array(imgs_batch)
    h_batch = np.array(h_batch)
    y_batch = np.array(y_batch, np.float32)
    
    x_batch = (imgs_batch, h_batch)

    return x_batch, y_batch

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

