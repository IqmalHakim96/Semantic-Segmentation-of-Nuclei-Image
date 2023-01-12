# Semantic-Segmentation-of-Nuclei-Image
 
## Introduction
The purpose of the project is to detect cell nuclei using image segmentation process on bimedical images. In this project, the U-net architecture was used to trained for image segmentation. The aim was to correctly segment the portion where the nuclei were seen. 

## Methodology

 ![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
 ![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)
 ![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
 ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
 ![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
 
 ### About dataset
The dataset folder contain train and test folder separately. Each folder are further divided into images and masks folder. Images folder contains the raw images which will be used as an input and masks folder contain the mask of image which are used for output or label data.

The input images need undergo preprocessing process by using feature scaling and same with labels convert to 0 and 1 binary values. The data was split into 80:20 ratio for training and testing process. 

### Model Architecture
As mentioned before, this projcet use U-net architecture method. Then, Transfer Learning was applied for deep learning by using MobileNet_V2. The documentation about U-net and MobileNet_V2 can refer to Tensorflow website. Image shown the model architecture that used in this project 

![model](https://user-images.githubusercontent.com/105650253/212072773-fd8c598b-c06d-43bb-a6dc-6bf9be86e463.png)

Adam optimizer was used and was evaluated on the sparse crossentropy loss function. After training the model for 10 epochs, the training accuracy was around 97% with loss 0.0608.

## Result
The model was evaluated with test data which is shown in figure below. The accuracy of this model is 97% and the actual output masks and predicted masks are shown in the figures below.

![result1](https://user-images.githubusercontent.com/105650253/212074346-9b95cefe-ae64-4dd5-9a0c-0391b9dbc9a4.png)
![result4](https://user-images.githubusercontent.com/105650253/212074380-f02ed2cd-2623-4cf1-9e1e-8abf7045a493.png)


## Discussion
1. The model are able to segmenting the nuclei with high accuracy and can deploy to more dataset. 

## Acknowledgement

The dataset are from 2018 Data Science Bowl from Kaggle and get get at this link : https://www.kaggle.com/competitions/data-science-bowl-2018/overview


 
