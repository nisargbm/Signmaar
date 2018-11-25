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



#gdrive/My Drive/Colab Notebooks/Dataset
#Dataset_Signature_Final.zip
train_path = 'gdrive/My Drive/Colab Notebooks/User 2'
valid_path = 'gdrive/My Drive/Colab Notebooks/Test'
#test_path  = ''

train_batches = ImageDataGenerator(rescale = 1./255).flow_from_directory(train_path,target_size=(224,224),classes=['real','forge'],batch_size = 16)
valid_batches = ImageDataGenerator(rescale = 1./255).flow_from_directory(valid_path,target_size=(224,224),classes=['real','forge'],batch_size = 8)

vgg16_model = k.applications.vgg16.VGG16(weights = 'imagenet',include_top = True)

#Add a layer where input is the output of the  second last layer 
x = Dense(2, activation='softmax', name='predictions')(vgg16_model.layers[-2].output)

#Then create the corresponding model 
model = Model(input=vgg16_model.input, output=x)

for layer in model.layers:
   layer.trainable = False
model.summary() 
model.get_layer('predictions').trainable = True  
model.summary()

adam = Adam(lr = 0.0001)
model.compile(loss=k.losses.categorical_crossentropy,optimizer = adam,metrics=['accuracy'])
model.fit_generator(train_batches,steps_per_epoch = 36,validation_data= valid_batches,validation_steps = 20, epochs = 30 , verbose = 1)
model.save('gdrive/My Drive/Colab Notebooks/signature_model_30epochs.h5')