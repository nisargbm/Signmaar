import os
import glob
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing import image
# import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from keras.models import Model

import h5py

model_file_path = 'gdrive/My Drive/Colab Notebooks/signature_model_50epochs.h5'
train_path = 'gdrive/My Drive/Colab Notebooks/User 2'
features_path = 'gdrive/My Drive/Colab Notebooks/features2.h5'
labels_path = 'gdrive/My Drive/Colab Notebooks/labels2.h5'

def load_my_model(file_path):
  new_model = load_model(file_path)
  model = Model(input=new_model.input, output=new_model.get_layer('fc1').output)
  return model

def get_features_labels(train_path, model):
  
  train_labels = os.listdir(train_path)
  le = LabelEncoder()
  le.fit([tl for tl in train_labels])
  features = []
  labels = []
  count = 1
  image_size = (224,224)
  for i,label in enumerate(train_labels):
    cur_path = train_path + "/" + label
    count = 1
    for image_path in glob.glob(cur_path+"/*.png"):
      img = image.load_img(image_path,target_size = image_size)
      x = image.img_to_array(img)
      x = np.expand_dims(x, axis=0)
      x = x/255
      feature = model.predict(x)
      flat = feature.flatten()
      features.append(flat)
      labels.append(label)
      print("[INFO]processed - "+str(count))
      count += 1
      return features, labels

def save_features_labels(features, features_path, labels, labels_path):
  le = LabelEncoder()
  le_labels = le.fit_transform(labels) 
  
  print ("[STATUS] training labels: {}".format(le_labels))
  print ("[STATUS] training labels shape: {}".format(le_labels.shape))
  h5f_data = h5py.File(features_path, 'w')
  h5f_data.create_dataset('dataset_1', data=np.array(features))

  h5f_label = h5py.File(labels_path, 'w')
  h5f_label.create_dataset('dataset_1', data=np.array(le_labels))

  h5f_data.close()
  h5f_label.close()

#model = load_my_model(model_file_path)
#features, labels = get_features_labels(train_path, model)
#save_features_labels(features, features_path, labels, labels_path)