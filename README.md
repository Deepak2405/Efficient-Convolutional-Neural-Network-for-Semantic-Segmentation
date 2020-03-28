# Efficient-Convolutional-Neural-Network-for-Semantic-Segmentation

This code is inspired from paper "ENet: A Deep Neural Network Architecture for Real-Time Semantic Segmentation" by Adam Paszke, Abhishek Chaurasia, Sangpil Kim, Eugenio Culurciello

Dataset : CamVid Dataset
Library : Keras for Deep Learning, OpenCV for Reading/writing images

Batchsize :  Standard Batchsize is 10, Set to 1 ( my PC  has 8GB RAM , multiple batchsize does not run in my current PC settings)

Current PC Settings
-------------------
GPU: RTX2080Ti
RAM : 8GB



Command to train the model
-------------------------
python3 trainEnet.py

trained Data saved as ENet.h5
Command to test the model
-------------------------
python3 predict.py

Prediction on CamVid standard video

