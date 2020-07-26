# Skin-Cancer-Classifier
CNN pre-trained model to visually diagnose between 3 types of skin lesions

This "Dermatologist-ai" project is part of my [Deep Learning Nanodegree with Udacity](https://www.udacity.com/course/deep-learning-nanodegree--nd101). The skin cancer classification model was trained and tested on google colab.

## Description
This project can visually diagnose between 3 types of skin lesions: melanoma, the deadliest form of skin cancer, and two types of benign lesions (nevi and seborrheic keratoses).

The data and objective are pulled from the [2017 ISIC Challenge on Skin Lesion Analysis Towards Melanoma Detection](https://challenge.kitware.com/#challenge/583f126bcad3a51cc66c8d9a). As part of the challenge, participants were tasked to design an algorithm to diagnose skin lesion images as one of three different skin diseases (melanoma, nevus, or seborrheic keratosis).

![](asset/skin_disease_classes.png)

I used 3 pre-trained models (VGG19, Inception-V3, ResNet152) to benefit from transfer learning. I adjusted the classification end of the network to the task at hand (classification between 3 labels).  I used the training and validation data to train a model that can distinguish between the three different image classes.
Then, the test images are used to gauge the performance of the model.

## Getting the Results
Once you have trained your model, create a CSV file to store your test predictions. Your file should have exactly 600 rows, each corresponding to a different test image, plus a header row. You can find an example submission file (`sample_submission.csv`) in the repository.

Your file should have exactly 3 columns:
- `Id` - the file names of the test images (in the same order as the sample submission file)
- `task_1` - the model's predicted probability that the image (at the path in Id) depicts melanoma
- `task_2` - the model's predicted probability that the image (at the path in Id) depicts seborrheic keratosis

Once the CSV file is obtained, you will use the `get_results.py` file to score your submission and obtain the scores in the three categories. This can be performed in the notebook. It also provides the corresponding ROC curves, along with the confusion matrix corresponding to melanoma classification.

## Dependencies

## Content

## Getting started
1.	Clone the repository and create a `data/` folder to hold the dataset of skin images.
2.	Create folders to hold the training, validation, and test images.
3.	Download and unzip the [training data](https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/skin-cancer/train.zip) (5.3 GB).
4.	Download and unzip the [validation data](https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/skin-cancer/valid.zip) (824.5 MB).
5.	Download and unzip the [test data](https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/skin-cancer/test.zip) (5.1 GB).
6.	Place the training, validation, and test images in the ` data/` folder, at `data/train/`, `data/valid/`, and `data/test/`, respectively. Each folder should contain three sub-folders (`melanoma/`, `nevus/`, `seborrheic_keratosis/`), each containing representative images from one of the three image classes.

I developed and run my notebook on Google Colab. To access the data, I mount my google drive as a first step in the notebook.


## Results
