import os
import glob
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing import image
import numpy as np
from keras.models import load_model
from keras.models import Model
import h5py
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import numpy as np
import h5py
import os
import json
import pickle
# import matplotlib.pyplot as plt

def model_pickle_creator(feature,label,customer_id,model_save_path):

	features = feature
	labels = label

	seed = 9
	test_size = 0.30


	# verify the shape of features and labels
	print ("[INFO] features shape: {}".format(features.shape))
	print ("[INFO] labels shape: {}".format(labels.shape))

	print ("[INFO] training started...")
	# split the training and testing data
	(trainData, testData, trainLabels, testLabels) = train_test_split(np.array(features),
																np.array(labels),
																test_size=test_size,
																random_state=seed)

	print ("[INFO] splitted train and test data...")
	print ("[INFO] train data  : {}".format(trainData.shape))
	print ("[INFO] test data   : {}".format(testData.shape))
	print ("[INFO] train labels: {}".format(trainLabels.shape))
	print ("[INFO] test labels : {}".format(testLabels.shape))

	# use logistic regression as the model
	print ("[INFO] creating model...")
	model = LogisticRegression(random_state = seed)
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
	print("Rank-1", format(rank_1))

	# evaluate the model of test data
	preds = model.predict(testData)

	# write the classification report to file
	#f.write("{}\n".format(classification_report(testLabels, preds)))
	print(format(classification_report(testLabels, preds)))
	#f.close()
	classifier_path = model_save_path + '/svm_user' + customer_id + '.pickle'
	print(classifier_path)
	pickle.dump(model, open(classifier_path, 'wb'))

def feature_extractor_according_to_user(cust_id, target_path, model_save_path, model_path):

	print("customer_id : ", cust_id, " -- model_save_path : ", model_save_path, " -- train_path : ", target_path)

	new_model = load_model(model_path)

	model = Model(input = new_model.input, output = new_model.get_layer('fc1').output)

	train_path = target_path
	train_labels = os.listdir(train_path)
	le = LabelEncoder()
	le.fit([tl for tl in train_labels])
	features = []
	labels = []
	count = 1
	image_size = (224, 224)
	customer_id = cust_id
	for i,label in enumerate(train_labels):
		cur_path = train_path + "/" + label
		count = 1
		list_of_customer_signatures = glob.glob(cur_path + "/*" + customer_id + ".png")
		if len(list_of_customer_signatures) > 0:
			for image_path in list_of_customer_signatures:
				img = image.load_img(image_path,target_size = image_size)
				x = image.img_to_array(img)
				x = np.expand_dims(x, axis=0)
				x = x/255
				feature = model.predict(x)
				flat = feature.flatten()
				features.append(flat)
				labels.append(label)
				print("image_path : " + image_path)
				print("[INFO]processed - " + str(count))
				count += 1

			#plt.imshow(x)
			#print("Info completed label - "+label)
	le = LabelEncoder()
	le_labels = le.fit_transform(labels) 
	labels = le_labels 
	features = np.array(features)
	model_pickle_creator(features, labels, customer_id, model_save_path)

