import numpy as np
import skimage.io as sk
from skimage.filters import gaussian
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