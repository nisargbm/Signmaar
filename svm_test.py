from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import numpy as np
import h5py
import os
import json
import pickle

features_path = 'gdrive/My Drive/Colab Notebooks/features2.h5'
labels_path = 'gdrive/My Drive/Colab Notebooks/labels2.h5'

h5f_data  = h5py.File(features_path, 'r')
h5f_label = h5py.File(labels_path, 'r')

features_string = h5f_data['dataset_1']
labels_string   = h5f_label['dataset_1']

features = np.array(features_string)
labels   = np.array(labels_string)

h5f_data.close()
h5f_label.close()

seed = 9
test_size = 0.30


# verify the shape of features and labels
print ("[INFO] features shape: {}".format(features.shape))
print ("[INFO] labels shape: {}".format(labels.shape))

print ("[INFO] training started...")
# split the training and testing data
(trainData, testData, trainLabels, testLabels) = train_test_split(np.array(features), np.array(labels), test_size=test_size, random_state=seed)

print ("[INFO] splitted train and test data...")
print ("[INFO] train data  : {}".format(trainData.shape))
print ("[INFO] test data   : {}".format(testData.shape))
print ("[INFO] train labels: {}".format(trainLabels.shape))
print ("[INFO] test labels : {}".format(testLabels.shape))

# use logistic regression as the model
print ("[INFO] creating model...")
model = LogisticRegression(random_state=seed)
model.fit(trainData, trainLabels)

print ("[INFO] evaluating model...")

rank_1 = 0

for (label, features) in zip(testLabels, testData):
  # predict the probability of each class label and
  # take the top-5 class labels
  #print(label,features)
  predictions = model.predict_proba(np.atleast_2d(features))[0]
  print("prediction_prob : ",predictions)
  predictions = np.argsort(predictions)[::-1][:2]
  if label == predictions[0]:
    rank_1 += 1
 

rank_1 = (rank_1 / float(len(testLabels))) * 100

#f.write("Rank-1: {:.2f}%\n".format(rank_1))
print("Rank-1",format(rank_1))

# evaluate the model of test data
preds = model.predict(testData)

# write the classification report to file
#f.write("{}\n".format(classification_report(testLabels, preds)))
print(format(classification_report(testLabels,preds)))
#f.close()
classifier_path = 'gdrive/My Drive/Colab Notebooks/svm_user2.pickle'
pickle.dump(model, open(classifier_path, 'wb'))