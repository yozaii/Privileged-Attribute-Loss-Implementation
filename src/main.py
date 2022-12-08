from model import *
import sys
import tensorflow as tf
import skimage.io as sk


if __name__ == '__main__':
    
    
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
    
    # tf.config.run_functions_eagerly(True)
    sys.path.append('../')
    sys.path.append('../data/RAFDB/raw/Image/aligned/')
    VGG16_WEIGHTS_PATH = '../pretrained_models/rcmalli_vggface_tf_notop_vgg16.h5'
    
    im_dir = '../data/RAFDB/raw/Image/aligned/'
    im_dir2 = 'data/RAFDB/raw/Image/aligned'
    h_dir = '../data/RAFDB/raw/landmarks/'
    
    dataset, dataset_test = load_keras_dataset_filepaths(im_dir, h_dir)
    dataset = dataset.batch(16)
    
    m = PALModel(weights_path = VGG16_WEIGHTS_PATH)     
    
    opt = keras.optimizers.Adam(learning_rate = 0.0005)
    loss_v = keras.losses.CategoricalCrossentropy()
    m.compile(loss = loss_v, optimizer=opt)
    
    m.fit(dataset_train)