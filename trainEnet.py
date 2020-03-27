'''
This code is to train enet semantic segmentation for CamVid Dataset.
Efficient Neural Network requires  low latency operation and  adopts ResNets archietecture.
##
'''
import  os 
import  glob 
import  numpy  as  np 
import  keras
from keras import regularizers
from keras.layers import Input
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from ENet import EfficientConvNet
from matplotlib import pyplot as plt
import dataset
import tensorflow as tf

height=360
width=480
channels=3
input_shape = (height, width, channels)
totalClass = 12
epochs = 100
batch_size = 1
log_filepath='./logs/'

data_shape = height*width

class_weighting = [0.2595, 0.1826, 4.5640, 0.1417, 0.5051, 0.3826, 9.6446, 1.8418, 6.6823, 6.2478, 3.0, 7.3614] #standard class weighting for segmentation


def main():
    print("loading data...")
    setOfData = dataset.Dataset(classes=totalClass)
    trainX, trainY = setOfData.load_data('train')   #uses 368 images for training

    trainX = setOfData.preprocess_inputs(trainX)    #preprocessing input data
    trainY = setOfData.reshape_labels(trainY)       #standard reshaping of labels for semantic segmentation
    

    testX, testY = setOfData.load_data('test')      #setting input and labels for test dataset
    testX = setOfData.preprocess_inputs(testX)
    testY = setOfData.reshape_labels(testY)
    trainAug = ImageDataGenerator(rotation_range=10, zoom_range=0.1,
                              horizontal_flip=True, rescale=1 / 255.0, fill_mode="nearest")

    valAug = ImageDataGenerator(rescale=1 / 255.0)
    trainAug.fit(trainX)
    valAug.fit(testX)
    print("creating model...")
        
    model = EfficientConvNet(totalClass)
    
    
    model.compile(loss="categorical_crossentropy", optimizer='adadelta', metrics=["accuracy"])
    H=model.fit_generator(trainAug.flow(trainX, trainY, batch_size=1),steps_per_epoch=len(trainX),  validation_data=(testX, testY),  epochs=100)

    model.save('Enet.h5')
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, epochs), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, epochs), H.history["val_acc"], label="val_acc")
    plt.title("Enet Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig("plot.png")

if __name__ == '__main__':
    main()



