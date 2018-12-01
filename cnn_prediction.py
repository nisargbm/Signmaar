from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import numpy as np

def predict (prediction_dataset_path, model_path = "./signature_model_50epochs.h5"):
  test_batches = ImageDataGenerator(rescale = 1./255).flow_from_directory(prediction_dataset_path, target_size=(224,224), classes=['real','forge'], batch_size = 60)
  test_imgs,test_labels = next(test_batches)
  test_labels = test_labels[:,0]
  new_model = load_model(model_path)
  prediction = new_model.predict_generator(test_batches, steps = 2, verbose = 1);
  print(np.round(prediction[:,0]))