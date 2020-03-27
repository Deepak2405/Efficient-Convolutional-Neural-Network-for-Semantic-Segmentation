import cv2
import numpy as np

from keras.applications import imagenet_utils

import  os

DataPath = './CamVid/'
data_shape = 360*480

class Dataset:
    def __init__(self, classes=12, train_file='train.txt', test_file='test.txt'):
        self.train_file = train_file
        self.test_file = test_file
        self.data_shape = data_shape
        self.classes = classes

    def normalized(self, rgb):
        #return rgb/255.0
        norm=np.zeros((rgb.shape[0], rgb.shape[1], 3),np.float32)

        b=rgb[:,:,0]
        g=rgb[:,:,1]
        r=rgb[:,:,2]

        norm[:,:,0]=cv2.equalizeHist(b)
        norm[:,:,1]=cv2.equalizeHist(g)
        norm[:,:,2]=cv2.equalizeHist(r)

        return norm

    def one_hot_it(self, labels):
        x = np.zeros([360,480,12])
        for i in range(360):
            for j in range(480):
                x[i,j,labels[i][j]] = 1
        return x

    def load_data(self, mode='train'):
        data = []
        label = []
        if (mode == 'train'):
            filename = self.train_file
        else:
            filename = self.test_file

        with open(DataPath + filename) as f:
            txt = f.readlines()
            txt = [line.split(' ') for line in txt]

        for i in range(len(txt)):
           # print((os.getcwd() + txt[i][0][7:]))
            data.append(self.normalized(cv2.imread(os.getcwd() + txt[i][0][7:])))
            label.append(self.one_hot_it(cv2.imread(os.getcwd() + txt[i][1][7:][:-1])[:,:,0]))
            print('.',end='')
        return np.array(data), np.array(label)


    def preprocess_inputs(self, X):
        return imagenet_utils.preprocess_input(X)

    def reshape_labels(self, y):
        return np.reshape(y, (len(y), self.data_shape, self.classes))
