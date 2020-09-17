# Skin-Cancer-Classifier
Use CNN model to visually diagnose between 3 types of skin lesions.

This "Dermatologist-ai" project is part of the [Deep Learning Nanodegree with Udacity](https://www.udacity.com/course/deep-learning-nanodegree--nd101). The skin cancer classification model was trained and tested using both own GPU and google colab.

## Description
This project can visually diagnose between 3 types of skin lesions: melanoma, the deadliest form of skin cancer, and two types of benign lesions, nevi and seborrheic keratoses.

The data and objective are pulled from the [2017 ISIC Challenge on Skin Lesion Analysis Towards Melanoma Detection](https://challenge.kitware.com/#challenge/583f126bcad3a51cc66c8d9a). As part of the challenge, participants were tasked to design an algorithm to diagnose skin lesion images as one of three different skin diseases (melanoma, nevus, or seborrheic keratosis).

![](asset/skin_disease_classes.png)

I started off with 3 pre-trained models (VGG16, Inception-V3, ResNet152) to benefit from transfer learning. These models have been already heaviliy trained for classification tasks using ImageNet database. ImageNet is a dataset of over 15 millions labeled high-resolution images with around 22,000 categories. To train these models, ILSVRC uses a subset of ImageNet of around 1000 images in each of 1000 categories. In all, there are roughly 1.2 million training images, 50,000 validation images and 100,000 testing images.

I adjusted the classification end of these networks to the task at hand (classification between 3 labels only).  I used the training and validation data to train a model that can distinguish between the three different image classes, saving the best parameters based on the reduction of the loss on the validation data.
Then, the test images are used to gauge the performance of the model on unseen images.

The proposed training, validation and test sets contain 2000, 160, 600 high-res RGB images respectively. The distribution between classes in the train set is heterogeneous. Nevus is highly over-represented (4 x to 5x more images than the other two classes). I therefore used [ISIC database](https://www.isic-archive.com/#!/topWithHeader/onlyHeaderTop/gallery?filter=%5B%5D) to augment the number of under represented classes. I added about 1,160 new training images which i could find for Melanoma and seborrheic keratosis. The training dataset had the following final composition:

![](asset/train_set.jpg)

Due to the large class imbalance in the training set (nevus is significantly over-represented), i tried to compensate by calculating weights to provide to the loss function during training. This did not prove efficient and penalized the overall accuracy of the model. So I did not use this option in the final runs.

I used various data augmentation techniques. Available images were of multiple sizes and reshaped to 299x299 (Inception) or 224x224 (other models). Inception V3's auxillary output was used.

## Getting the Results
Once you have trained your model, create a CSV file to store your test predictions. Your file should have exactly 600 rows, each corresponding to a different test image, plus a header row. You can find an example submission file (`sample_submission.csv`) in the repository.

Your file should have exactly 3 columns:
- `Id` - the file names of the test images (in the same order as the sample submission file)
- `task_1` - the model's predicted probability that the image (at the path in Id) depicts melanoma
- `task_2` - the model's predicted probability that the image (at the path in Id) depicts seborrheic keratosis

Once the CSV file is obtained, the notebook provides the scores in the three categories. It also provides the corresponding ROC curves, along with the confusion matrix corresponding to melanoma classification. Note that you can also use the `get_results.py` file to score your submission.

## Content
- Mount Google Drive
- Select folder with notebook and data folder in google colab
- Download the datasets and unzip into data folder
- import librairies and count available samples
- explore the data
- calculate corrective weights to compensate for skewness
- Set Dataloader
- Define train, validate and test functions
- Train the models
    - Obtain pre-trained VGG16 and adjust classifier section
    - specify loss function and optimizer
    - train and test the model
   - Repeat with Inception V3 and ResNet-152
 - Get predictions and prepare results file
 - Evaluate predictions and plot scores, ROC and confusion matrix

## Getting started
1.	Clone the repository and create a `data/` folder to hold the dataset of skin images.
2.	Create folders to hold the training, validation, and test images.
3.	Download and unzip the [training data](https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/skin-cancer/train.zip) (5.3 GB).
4.	Download and unzip the [validation data](https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/skin-cancer/valid.zip) (824.5 MB).
5.	Download and unzip the [test data](https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/skin-cancer/test.zip) (5.1 GB).
6.	Place the training, validation, and test images in the ` data/` folder, at `data/train/`, `data/valid/`, and `data/test/`, respectively. Each folder should contain three sub-folders (`melanoma/`, `nevus/`, `seborrheic_keratosis/`), each containing representative images from one of the three image classes.

I developed and run my notebook on Google Colab. To access the data, I mount my google drive as a first step in the notebook.


## Results
