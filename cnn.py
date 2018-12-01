import numpy as np
import keras as k
from keras.models import Sequential
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Conv2D, Dropout
from keras.models import Sequential
from keras.layers.core import Dense,Flatten
from keras.optimizers import adam
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras import optimizers
from keras.utils.data_utils import get_file
# import matplotlib.pyplot as plt

def VGG16(filepath, input_shape=None , classes=1000):
    model = Sequential()
    # Block 1
    model.add(Conv2D(64, (3, 3), input_shape=input_shape, activation='relu', padding='same', name='block1_conv1'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

    # Block 2
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

    # Block 3
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

    # Block 4
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))

    # Block 5
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))

    # Classification block
    model.add(Flatten(name='flatten'))
    model.add(Dense(4096, activation='relu', name='fc1'))
    model.add(Dense(4096, activation='relu', name='fc2'))
    model.add(Dense(classes, activation='softmax', name='predictions'))

    model.load_weights(filepath, by_name=True)
    model.summary()
    return model

def my_model(VGG_model, classes=2):
    x = Dense(2, activation='softmax', name='predictions')(vgg16_model.layers[-2].output)
    model = Model(input=vgg16_model.input, output=x)
    for layer in model.layers:
      layer.trainable = False
    model.get_layer('fc1').trainable = True
    model.get_layer('fc2').trainable = True
    model.get_layer('predictions').trainable = True
    model.summary()
    return model

def load_train_batches(train_path, target_width, target_height, batch_size, color_mode = 'rgb'):
    train_batches = ImageDataGenerator().flow_from_directory(train_path, target_size=(target_width, target_height), color_mode = color_mode, classes=['real','forge'], batch_size = batch_size)
    return train_batches

def load_val_batches(valid_path, target_width, target_height, batch_size, color_mode = 'rgb'):
    valid_batches = ImageDataGenerator().flow_from_directory(valid_path, target_size=(target_width, target_height), color_mode = color_mode, classes=['real','forge'], batch_size = batch_size)
    return valid_batches

def train_cnn_model(cnn_weights_filepath, input_shape, train_dataset_path, valid_dataset_path):

    train_batches = load_train_batches(train_path, target_width=224, target_height=224, batch_size=16)
    valid_batches = load_val_batches(valid_path, target_width=224, target_height=224, batch_size=8)
    #Get Weight file
    weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels.h5', filepath)

    #Build Vgg model with weights
    vgg16_model = VGG16(filepath = weights_path, input_shape=input_shape)

    #build model for classification
    model = my_model(vgg16_model)

    gd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    adam = Adam(lr = 0.0001)
    model.compile(loss=k.losses.categorical_crossentropy,optimizer = gd,metrics=['accuracy'])

    history = model.fit_generator(train_batches, steps_per_epoch=36, validation_data=valid_batches, validation_steps=20, epochs=50, verbose=1)

    # UNCOMMENT TO VIEW VISUALIZATION OF PERFORMANCE BY LOSS MEASURE
    # UNCOMMENT matplotlib in the headers
    #
    # loss = history.history['loss']
    # val_loss = history.history['val_loss']
    # epochs = range(1, len(loss) + 1)
    # plt.plot(epochs, loss, color='red', label='Training loss')
    # plt.plot(epochs, val_loss, color='green', label='Validation loss')
    # plt.title('Training and validation loss')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.show()

    # UNCOMMENT TO VIEW VISUALIZATION OF PERFORMANCE BY ACCURACY
    # UNCOMMENT matplotlib in the headers
    #
    # acc = history.history['acc']
    # val_acc = history.history['val_acc']
    # plt.plot(epochs, acc, color='red', label='Training acc')
    # plt.plot(epochs, val_acc, color='green', label='Validation acc')
    # plt.title('Training and validation accuracy')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.show()

    return model
    


