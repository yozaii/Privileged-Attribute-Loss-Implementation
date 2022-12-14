from model import *
import sys
import tensorflow as tf
import skimage.io as sk


if __name__ == '__main__':
    
    
    # =================== DATA PREPARATION =================== #
    
    # Add root directory to path
    sys.path.append('../')
    
    # Load model weights
    VGG16_WEIGHTS_PATH = '../pretrained_models/rcmalli_vggface_tf_notop_vgg16.h5'
    PAL_MODEL_WEIGHTS_PATH = '../pretrained_models/PAL_model/checkpoint'
    
    # Dataset directories (images / heatmaps)
    im_dir = '../data/RAFDB/raw/Image/aligned/'
    h_dir = '../data/RAFDB/raw/landmarks/'
    
    # Load dataset filepaths and pass them to the generator
    train_dataset_partition, test_dataset_partition = load_partition()
    training_generator = DataGenerator(train_dataset_partition)
    
    # =================== MODEL BUILDING ====================== #
    
    # Create model
    m = PALModel(weights_path = VGG16_WEIGHTS_PATH)     

    # Model compilation params
    opt = keras.optimizers.Adam(learning_rate = 0.0005)
    loss_fn = keras.losses.CategoricalCrossentropy()
    met = [keras.metrics.CategoricalAccuracy(name='categorical_accuracy')]
    
    # compile model
    m.compile(loss = loss_fn, optimizer=opt, metrics = met)
    
    # callbacks for training
    checkpoint_filepath = '../pretrained_models/PAL_model/checkpoint'
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='categorical_accuracy',
        mode='max',
        save_best_only=True)
    
    # train
    m.fit(training_generator, epochs = 2, steps_per_epoch=2,
          callbacks=[model_checkpoint_callback])
