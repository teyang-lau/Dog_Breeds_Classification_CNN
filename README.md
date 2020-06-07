# Dog Breeds Classification with CNN and Transfer Learning

[![made-with-kaggle](https://img.shields.io/badge/Made%20with-Kaggle-lightblue.svg)](https://www.kaggle.com/)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-blue.svg)](https://www.python.org/)
[![made-with-Markdown](https://img.shields.io/badge/Made%20with-Markdown-1f425f.svg)](http://commonmark.org)
[![Generic badge](https://img.shields.io/badge/STATUS-COMPLETED-<COLOR>.svg)](https://shields.io/)
[![GitHub license](https://img.shields.io/github/license/teyang-lau/Dog_Breeds_Classification_CNN.svg)](https://github.com/teyang-lau/Dog_Breeds_Classification_CNN/blob/master/LICENSE)

Author: TeYang, Lau <br>
Last Updated: 7 June 2020

<img src = './Pictures/dogbreeds.jpg'>

### **Please refer to this [notebook](https://www.kaggle.com/teyang/dog-breeds-classification-using-transfer-learning?scriptVersionId=35621180) on Kaggle for a more detailed description, analysis and insights of the project.** 

## **Project Motivation** 
This is my first deep learning project after taking a course about it. I chose a dataset that has much diveristy yet simple for a first dive into the world of neural networks. Instead of using the common cats or dogs dataset, I chose this as there are multiple classifications/dog breeds, making it a bit more challenging than binary classification.

## **Project Goals** 
1. To build a deep learning model to **classify** pictures of 120 different dog breeds
2. Use data augmentation to generate more diversity in the pictures

## **Project Overview** 
* Resizing images to fit into model
* Data augmentation to create diversity
* Using transfer learning to pre-train model
* Tuning hyperparameters of the model such as learning rate, min-batch size etc
* Evaluating training and validation set to achieve the best model performance

## **About this dataset** 
This [Stanford Dog Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/) is shared by courtesy of Stanford. It contains 20580 pictures of dogds consisting of 120 different breeds. For this analysis, I only limit my train, validation and test sets to only 4000 images. After splitting, only 2560 images were used for training, and the rest split between validation and testing. The reason was because of lack of hardware memory to store such a large array of images. On hindsight and after learning more about dealing with large datasets, there is actually a way to get around this problem which involves only loading the data when it is needed for training.

## **Dog Breeds Samples** 
Here are some sample pictures of the dogs and their associated breeds. Notice that some pictures include humans while in others, the dog may be quite small. This adds to the difficulty but if the model is trained correctly, it should be able to correctly classify these images, which are usually quite prevalent in the real world.

<img src = './Pictures/samples.png'>

## **Data Augmentation** 
I performed data augmentation by transforming the pictures (rotation, flipping, horizontal and vertical shifting) to produce more diversity in the dataset. 

<img src = './Pictures/augmentation.png'>

## **Model Architecture** 
I made use of the power of transfer learning to pretrain the model. As these were already pre-trained, we can use the first layers of weights and add a few additional layers to classify the images into 120 categories. For this project, I used Google's Inception V3 model.

<img src = './Pictures/model_architecture.png'>


## **Model Training Performance** 
I tuned a few hyperparameters of the model such as the learning rate, mini-batch size and algorithm used. From the plot below, we can see that the training is still slightly underfitting but is starting to converge at 20 epochs. 

<img src = './Pictures/evaluation.png'>

## **Model Predictions** 
Our model achieved an accuracy of **~86%-88%** and this is not bad considering that we trained it on only 2650 images. Below are samples of the predictions as well as the probability of its predictions. 

<img src = './Pictures/predictions.png'>

## **Difficulties Faced** 
* Hardware memory limitations prevented me from loading the entire dataset into 1 single array for training. However, there is a function in Keras called flow_from_directory which allows the loading of files only when they are needed. Future work might involve learning and exploring this method to solve this limitation.

* Understanding what is going inside the neural network black box is something very difficult as there are just too many computations running in conjunction and in cycles. I believe that as neural networks receive more research, understanding of them will become better.

* For this project, I used Kaggle's GPU to speed up the model training process which I could not have done it on my local desktop. As big data is becoming more prevalent, understanding and learning how to use cloud services to run big projects will be an important skill to acquire.

## **Conclusions** 
Convolutional neural networks are really powerful models to classify images. With enough data from health-related problems, CNNs can be used to train models from pictures of healthcare tests such as X-Ray and MRI scans to detect diseases. Although healthcare practitioners are experts in diagnosing diseases, human error is still possible, and machine learning models can help reduce them and might also potentially find new ways/patterns in disease diagnostics.



