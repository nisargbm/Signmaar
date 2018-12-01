import cnn
import svm
import cnn_prediction
import os
import glob

'''
###########################
# CNN Training Phase
###########################

# Input for CNN training

input_shape = (224, 224, 3)
cnn_train_dataset_path = './Training'
cnn_valid_dataset_path = './Test'

# CNN ImageNet weight file

cnn_weights_filepath = "https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5"
cnn_model_save_path = './signature_model_50epochs.h5'


# Train CNN and save model

model = cnn.train_cnn_model(cnn_weights_filepath = cnn_weights_filepath,
							input_shape = input_shape, 
							train_dataset_path = cnn_train_dataset_path, 
							valid__dataset_path = cnn_valid_dataset_path)
model.save(model_save_path)

'''


##############################
# Predictions using only CNN 
##############################

cnn_saved_model_path = './signature_model_50epochs.h5'
test_data_path = './dataset1'

cnn_prediction.predict(test_data_path, cnn_saved_model_path)




##############################
# SVM for classification Phase
##############################

cnn_saved_model_path = './signature_model_50epochs.h5'

customer_id_list = ["001","002","003","004","005"]
model_save_path = "./features_path"
train_path = './dataset1'

if not os.path.exists(model_save_path):
	os.makedirs(model_save_path)

for i in customer_id_list:
	svm.feature_extractor_according_to_user(i,train_path,model_save_path,cnn_saved_model_path)

