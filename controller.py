import cnn
import cnn_prediction


input_shape = (224, 224, 3)
cnn_train_dataset_path = 'gdrive/My Drive/Colab Notebooks/Training'
cnn_valid_dataset_path = 'gdrive/My Drive/Colab Notebooks/Test'
cnn_weights_filepath = "https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5"
model_save_path = 'gdrive/My Drive/Colab Notebooks/signature_model_vgg_impl_50epochs.h5'

model = cnn.train_cnn_model(cnn_weights_filepath=cnn_weights_filepath, input_shape=input_shape, train_dataset_path=cnn_train_dataset_path, valid__dataset_path=cnn_valid_dataset_path)
model.save(model_save_path)


