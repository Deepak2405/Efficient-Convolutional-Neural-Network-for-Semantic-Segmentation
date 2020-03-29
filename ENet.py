'''
Code for designing efficient convolutional neural network for semantic segmentation.
Keras library is used for designing this CNN.
ENet uses the principle of ResNet archietecture by having 1x1 bottleneck convolution at both encoder and decoder side.
The database for training used is CamVid. All the frames are resized to 480x360x3 and total classes are 12.
'''
from keras.layers.core import Activation, Reshape
from keras.layers.normalization import BatchNormalization
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
