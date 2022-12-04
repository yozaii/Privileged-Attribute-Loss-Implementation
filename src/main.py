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
    
    train_x, train_y = load_train_dataset()
    
    
    m = PALModel(weights_path = VGG16_WEIGHTS_PATH)     
    m.compile()
    m.summary()
    
    
    # im = np.ones(224,224,3)