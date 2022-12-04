import numpy as np
from tensorflow import keras
import skimage.io as sk
from skimage.filters import gaussian
from skimage.util import img_as_float
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
# ================ DATA LOADING FUNCTIONS =============== #
# ======================================================= #



def load_partition(filepath = None):

    # To use global variable if no filepath is given
    if filepath == None:
        filepath = PARTITION_FILEPATH
        
    # Open file and read all lines
    f = open(filepath, 'r')
    Lines = f.readlines()
    
    # list for data (ex : test_aligned_0001), and its label (0-6)
    data = list()
    label = list()
    
    for line in Lines:
        
        
        
        # x and y coordinates of landmarks
        x, y = line.split()
        
        # Remove file extension and append to list
        x = x.replace('.jpg','')
        
        data.append(x)
        label.append(y)
        
    return data, label


def load_heatmap(dataname, im_h, im_w, sigma, dirname = None):
    
    if dirname == None:
        dirname = LANDMARKS_DIRPATH
        
    filename = dataname + '_aligned.txt'
    filename = dirname + '/' + filename
    
    
    # Open file and read all lines
    f = open(filename, 'r')
    Lines = f.readlines()
    
    # List of heatmaps for each landmark
    heatmap = np.zeros((im_h,im_w), dtype = np.float)
    
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

def load_all_landmark(dirname, im_h, im_w):
    
    # list that will hold trainset / testset heatmaps
    train_hm = list()
    test_hm = list()
    
    for filename in os.listdir(dirname):

        f = os.path.join(dirname, filename)
        heatmap = load_landmarks_file(f, im_h, im_w, 3)
        
        # filename's second letter is r -> train heatmap
        if filename[1] == 'r':
            train_hm.append(heatmap)
            
        else :
            test_hm.append(heatmap)
    
    return np.array(train_hm), np.array(test_hm)

def load_img(filename):
    pass

def load_imgs():
    pass

def load_label():
    pass

def load_labels():
    pass

def load_data():
    pass
    # return (train_imgs, train_labels), (test_imgs, test_labels)

# ======================================================= #
# =============== DATA PROCESSING FUNCTIONS ============= #
# ======================================================= #

def img_resize(img, width, height):
    pass


def load_partition(filepath = None):


    # To use global variable if no filepath is given
    if filepath == None:
        filepath = PARTITION_FILEPATH
        
    # Open file and read all lines
    f = open(filepath, 'r')
    Lines = f.readlines()
    
    # list for data (ex : test_aligned_0001), and its label (0-6)
    data = list()
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
        
        data.append(x)
        
    f.close()
        
    return data, (np.array(train_label), np.array(test_label))

def load_heatmap(filename, im_h, im_w, sigma = 3):

    # Open file and read all lines
    f = open(filename, 'r')
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

def load_img(filename):
    
    img = sk.imread(filename)
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

# ======================================================= #
# =============== DATA PROCESSING FUNCTIONS ============= #
# ======================================================= #

def img_resize(img, width, height):
    pass

def create_augmented_img():
    pass


def process_imgs(imgs):
    # img_resize(img, width, height)
    pass
    

def process_prior_heatmaps(imgs):
    pass



    
if __name__ == '__main__':
    
    
    sys.path.append('../')
    

    filename = '../data/RAFDB/raw/landmarks/test_0006_aligned.txt'
    dirname = '../data/RAFDB/raw/landmarks'
    
    
    # ====================================================== #
    # start time
    start = time.time()
    
    for i in range(12000):
        im = load_heatmap('test_0006', 112, 112, 3)
    # train_hm, test_hm = load_all_landmark_files(dirname, 112, 112)
    
    # end time
    end = time.time()
    
    elapsed = end - start
    print(elapsed)
    
    sk.imshow(im)
    # print(train_hm.shape, test_hm.shape)

    # filename = '../data/RAFDB/raw/landmarks/test_0006_aligned.txt'
    # dirname = '../data/RAFDB/raw/landmarks'
    
    
    # ====================================================== #
    # # start time
    # start = time.time()
    
    # for i in range(16000):
    #     im = load_heatmap('../data/RAFDB/raw/landmarks/test_0006_aligned.txt', 112, 112, 3)
    # # train_hm, test_hm = load_all_landmark_files(dirname, 112, 112)
    
    # # end time
    # end = time.time()
    
    # elapsed = end - start
    # print(elapsed)
    
    # sk.imshow(im)
    # # print(train_hm.shape, test_hm.shape)

    # ====================================================== #

    # # start time
    # start = time.time()
    
    # x, y = load_partition()
    # heatmaps = list()
    
    # for l in x:
    #     heatmap = load_landmarks(l, 112, 112, 3)
    #     heatmaps.append(heatmap)
        
    # # end time
    # end = time.time()

    
    # elapsed = end - start
    # print(elapsed)

    
    # elapsed = end - start
    # print(elapsed)
    
    # ====================================================== #
    
    # start time
    start = time.time()
    
    data, y = load_partition()
    train_y, test_y = y
    train_y = keras.utils.to_categorical(train_y)
    test_y = keras.utils.to_categorical(test_y)
    
    
    
    img_filenames = list()
    heatmap_filenames = list()
    
    for dataname in data:
        img_filenames.append(dataname + '_aligned.jpg')
        heatmap_filenames.append(dataname + '_aligned.txt')
        
    print('after filenames', time.time() - start)
    
    img_dir = '../data/RAFDB/raw/Image/aligned/'
    heatmap_dir = '../data/RAFDB/raw/landmarks/'
      

    train_h, test_h = load_all_heatmaps(heatmap_dir, heatmap_filenames, 112, 112)
    print(time.time() - start)
    
    train_x, test_x = load_all_imgs(img_dir, img_filenames)
    
    print(time.time() - start)
    train_x = train_x.astype(np.float32)/255
    test_x = train_x.astype(np.float32)/255
    
    
    
    
    # end time
    end = time.time()
    
    elapsed = end - start
    print(elapsed)
    
    # ====================================================== #
    
    # filename = '../data/RAFDB/raw/Image/aligned/test_0006_aligned.jpg'
    # im = sk.imread(filename)
    # im = img_as_float(im)

