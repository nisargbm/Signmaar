# Signmaar: A signature verification system

A system for extracting features like strokes, curves, dots, dashes, writing fluidity & style using deep learning algorithms like CNN in a writer independent manner and then use the CNN model to act as a feature extractor for training writer dependent SVM classifiers for each user

# Setup

## How to set up a Python development environment

A Python development environment is a folder which you keep your code in, plus a "virtual environment" which lets you install 
additional library dependencies for that project without those polluting the rest of your laptop.

    mkdir my-new-python-project
    cd my-new-python-project
    virtualenv --python=python3.6 venv
    # This will create a my-new-python-project/venv folder
    touch .gitignore
    subl .gitignore
    # Add venv to your .gitignore

Now any time you want to work on your project, you need to "activate your virtual environment". Do that like this:

    cd my-new-python-project
    source venv/bin/activate

You can tell it is activated because your terminal prompt will now look like this:

    (venv) computer:my-new-python-project user$

Now install all the necessary requirements for the project

    pip install -r requirements.txt
    
Because of size limit(300mb) the trained cnn model was not uploaded.Kindly download the model from [here](https://drive.google.com/open?id=1aItycfygSuqksetZEzAN9AC9Dy19tt85)

## Instructions to use

This project folder is spilt into 3 sections

CNN Training: For training CNN --> cnn.py
Predictions using only CNN --> cnn_prediction.py
Predictions using SVM classifier --> svm.py
These files can be accessed using controller.py file.

CNN training section has been commented. Other 2 sections are used for predictions.

The source code folder contains the relevant datasets. If any new datasets are added within the folder, kindly change the location path. All location paths can be changed in the controller.py file and no changes are required in any other file.

#### The source code is also available [here](https://drive.google.com/file/d/1EEL0HGg7W-9GCJLhn3mVZYBL5j-KD5I9/view?usp=sharing)
