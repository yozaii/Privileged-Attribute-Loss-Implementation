from model import *
import sys
import tensorflow as tf
import skimage.io as sk


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
    
    # Compilation parameters
    
    # # Polynomial decay
    # learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
    # starter_learning_rate,
    # decay_steps,
    # end_learning_rate,
    # power=0.5)
    opt = keras.optimizers.Adam(learning_rate = 0.0005)
    loss_fn = keras.losses.CategoricalCrossentropy()
    
    m.compile(loss = loss_fn, optimizer=opt)
    
    m.fit(dataset_train)