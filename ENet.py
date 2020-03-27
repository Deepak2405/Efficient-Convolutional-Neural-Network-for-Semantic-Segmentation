'''
Code for designing efficient convolutional neural network
'''
from keras.layers.core import Activation,SpatialDropout2D, Permute,  Reshape
from keras.layers.merge import add, concatenate
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.engine.topology import Input
from keras.models import Model
from encoder import initial_block,bottleneck,encoder
from decoder import decodeNeck,decoder


def EfficientConvNet(classes, inpHeight=360, inpWidth=480):
	img_input = Input(shape=(inpHeight, inpWidth, 3))
	Effnet = encoder(img_input)
	Effnet = decoder(Effnet, classes)
	output = Model(img_input, Effnet).output_shape
	Effnet = (Reshape((output[1]*output[2], classes)))(Effnet)
	Effnet = Activation('softmax')(Effnet)
	model = Model(img_input, Effnet)
	model.outputWidth = output[2]
	model.outputHeight = output[1]

	return model
