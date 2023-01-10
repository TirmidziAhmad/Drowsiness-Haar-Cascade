# Driver-Drowsiness-Detection-using-Computer-Vision
In this project image processing and deep learning technology specifically InceptionV3 transfer learning model and Haar Cascade Classifier have been used.

## Project Demonstration 
https://user-images.githubusercontent.com/69957858/211492369-d5526588-e5cd-4021-8335-0d9bbc743e2e.mp4

## Dataset
MRL Eye Dataset has been used. You can download the dataset from here - http://mrl.cs.vsb.cz/eyedataset
* Eye images of 37 candidates.
* Convert data from 37folders to train and test folders
* 80% of Data were used for Training
* 20% of Data were used for Testing and Evaluating 
<p align="center"> <img src="https://user-images.githubusercontent.com/69957858/211485836-a8c0498a-b28b-4885-be8b-f5e3501154a2.png" width="400" height="250">  
<img src="https://user-images.githubusercontent.com/69957858/211485863-2f8f05e9-e039-47c6-9a1e-fe33b059ff52.png" width="400" height="250"> </p>

<p align="center"><img src="https://user-images.githubusercontent.com/69957858/211485876-0856e326-61e8-410d-83c7-833ad1338ebe.png" width="400" height="100">
<img src="https://user-images.githubusercontent.com/69957858/211485892-258cf42e-ca63-44b4-adaf-be734dc40fa2.png" width="400" height="100"> </p>

## Python libraries and packages Used
* import tensorflow as tf
* from tensorflow.keras.applications import InceptionV3
* from tensorflow.keras.models import Model
* from tensorflow.keras.layers import Dropout,Input,Flatten,Dense,MaxPooling2D
* from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Data Augumentation
* from tensorflow.keras.models import load_model
* import cv2
* import numpy as np
* import time
* from pygame import mixer

## Project Objective
Our main goal is to detect driver drowsiness using facial features with the help of computer vision, image processing and machine learning. Features such as eye closure, blinking frequency, yawning, and head-nodding can be used to detect drowsiness. We are working with only eye state feature.


## Flowchart of the Project
<p align="center"><img src="https://user-images.githubusercontent.com/69957858/211485645-1db84450-d74a-49d1-b9d9-fc94bacc2743.png" width="600" height="600"> </p>

## Results
After 50epochs, the model training accuracy was 95.58%, validation accuracy was 90.53% and Test Accuracy was 85.98%. Whereas the training loss was 0.117, validation loss was 0.254 and testing loss was 0.52. The model can perform better if we train it with more datasets and run more epochs to achieve better accuracy. 

**Training vs Validation Accuracy for 50epochs**
<p align="center"><img src="https://user-images.githubusercontent.com/69957858/211487485-ae381b45-2808-4dc3-995b-776b638d2557.png" width="450" height="400"> </p>

**Training vs Validation Loss for 50epochs**
<p align="center"><img src="https://user-images.githubusercontent.com/69957858/211487498-c65f01f1-3731-4634-b570-343477341023.png" width="450" height="400"> </p>

**Confusion Matrix**
<p align="center"><img src="https://user-images.githubusercontent.com/69957858/211490520-6dcef6be-d6a2-495f-84a8-1bdfa197fc3e.png" width="600" height="500"> </p>

**Classification Report**
<p align="center"><img src="https://user-images.githubusercontent.com/69957858/211488449-28bec2c8-3141-495a-a0f2-da667be2ef5f.png" width="400" height="250"> </p>


