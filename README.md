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

## Authors

* **Nisarg Mistry**- [Nisarg](https://github.com/nisargbm)
* **Rahul Singh**- [Rahul](https://github.com/RahulSinghh)
* **Meet Shah**
* **Aditya Malshikhare**- [The Paired Electron](https://github.com/thePairedElectron)

## Future Work
We are planning to build a fully pledged cheque and DD processing framework which will be the global standard for advanced image analysis and intelligent recognition software used to seamlessly, precisely and securely process checks and other payment documentation by banks, financial institutions and other progressive corporations around the world. Our framework will focus on to be a pioneer that helps give greater confidence, freedom and funds accessibility to banks and its customers alike. By using this framework, consumers can enjoy a time-savings and more secure check processing benefits– which come in handy for today’s fast-paced society.

Applications: Information Verification, Secure Transactions and Fraud Detection

As our framework can be entrusted to help process financial information and transactions, it recognizes that three issues are paramount for our customers: making sure the extracted fields, including the courtesy (CAR) and legal (LAR) amounts, are correct, verifying the parties on each side of the transaction are legitimate by performing positive-pay functionality, and ensuring the transaction is completed quickly and securely. Capturing and recognizing the data found on each payment is a crucial and multi-dimensional capability that our framework will provide. In fact, it can be used during millions of successful transactions every day, helping to minimize errors and detect fraud within the global payment market.

Banks can shape their operations to utilize our framework as their core engine. From branches and teller windows, to back offices and central processes, ATMs, merchants, fraud applications and more – you can bank securely from almost anywhere in the world, and our framework will support these needs. Intelligent Word Recognition (IWR), ICR, OCR and handwriting analysis combined on one product make it possible.

## Key features that we dreamed about
 • Signature Verification
 
 • CAR / LAR Mismatch Detection
 
 • Post-Dated / Slate-Dated Check Detection
 
 • Positive Pay / Payee Name Verification
 
 • Black List Payee Name Verification
 
 • Rear Endorsement Detection
 
 • Cursive Handwritten Fields
 
 
 • MICR Code Recognition
 
 • Memo Line Recognition
 
 • Check Usability and Validity Tests
 
 • Payment Type Classification
 
 • Money Order Detection and Recognition
 
 • Document Identifier
 
 • Coupon Assisted Amount Decisioning
