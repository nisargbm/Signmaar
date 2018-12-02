from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import itertools


def predict (prediction_dataset_path = 'dataset1' , model_path = "signature_model_50epochs.h5"):
  
  new_model = load_model(model_path)
  batch_size = 20
  test_batches = ImageDataGenerator(rescale = 1./255).flow_from_directory(prediction_dataset_path,target_size=(224,224),classes=['real','forge'],batch_size =20)
  #test_labels = test_labels[:,0]
  data_list = []
  batch_index = 0
  while batch_index <= test_batches.batch_index:
      data,label = test_batches.next()
      data_list.append(label[:,0])
      batch_index = batch_index + 1
  total_num = batch_index * batch_size
  print(total_num)
  prediction = new_model.predict_generator(test_batches,steps = batch_index,verbose = 0);
  predictions = np.round(prediction[:,0])
 
  

  data_array = np.asarray(data_list)
  flat_list = [item for sublist in data_array for item in sublist]
  
  
  pred = []
  expect = []
  for i  in range(0,total_num):
    if(predictions[i]==0):
      pred.append('real')
    if(predictions[i]==1):
      pred.append('forge')
    if(flat_list[i]==0):
      expect.append('real')
    if(flat_list[i]==1):
      expect.append('forge')
      
  print(pred)
  print(expect)
  
  '''
  
  ################################
  #TO See the CONFUSION Matrix of The PREDICTED Value 
  #Kindly Run the Below code
  #################################3
  
  
  import matplotlib.pyplot as plt
  from sklearn.metrics import confusion_matrix
  y_test = flat_list
  y_pred = predictions
  cnf_matrix = confusion_matrix(y_test, y_pred)
  class_names = ['real','forge']
  plot_confusion_matrix(cnf_matrix, classes=class_names,
                       title='Confusion matrix CNN VGG architecture')
  '''
  
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
						  						  
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, CNN VGG architecture')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 4.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="black" if cm[i, j] > thresh else "white")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

	
predict()