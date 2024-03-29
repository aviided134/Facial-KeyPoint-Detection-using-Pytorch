# Facial-KeyPoint-Detection-using-Pytorch

In this project, I have created a deep learning based facial detection model using Convolution Neural Network(CNN). The main objective is to predict the location of 14 distinguishing Keypoints on each face. Facial keypoints include points around the eyes, nose, and mouth on the face. 
![image](https://github.com/aviided134/Facial-KeyPoint-Detection-using-Pytorch/assets/119523062/d4d0dad1-052a-4339-b173-71a2bb2cddbf)

Facial keypoints (also called facial landmarks) are the small magenta dots shown on each of the faces in the image above. In each training and test image, there is a single face and 14 keypoints, with coordinates (x, y), for that face. These keypoints mark important areas of the face: the eyes, corners of the mouth, the nose, etc.

# Dataset
Total data- 6000 images.
Train data - 3840
Test data = 1200
validation data = 960

 The information about the images and keypoints in this dataset are summarized in CSV files, which we can read in using pandas. For each data point(row) in CSV, it contains the image name, and 68 other columns describing the 38 distinguishing pairs of keypoints on the face. We can read the training CSV and get the annotations of 68 keypoints in an (N, 2) array where N is the number of keypoints(14) and 2 is the dimension of the keypoint coordinates (x, y).

# Transformation
Used albumentation to rotate, scale and shift the image. 

# Model
A total of 3 conv layers with kernel size of (3,3) and stride of 1 followed by a MaxPool2d layer. A ReLU function is used after each conv layer. At last, the model is flattened and Linear functio is used.
