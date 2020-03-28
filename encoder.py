from keras.layers.core import Activation,SpatialDropout2D, Permute,  Reshape
from keras.layers.merge import add, concatenate
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.engine.topology import Input
from keras.models import Model

'''
The input image has a resolution of 480 x 360 x3 .
The output size of the initial block of 16 x 256 x 256 by concatenating of the convolution (13 filters) and MaxPooling (2 x 2; without overlap)
'''
def initial_block(inputData, filterSize=13, nb_row=3, nb_col=3, strides=(2, 2)):
	conv = Conv2D(filterSize, (nb_row, nb_col), padding='same', strides=strides)(inputData)
	max_pool = MaxPooling2D()(inputData)
	merger = concatenate([conv, max_pool], axis=3)
	return merger
'''
Each branch consists of three convolutional layers.
As per the paper, Batch normalization and PReLU are placed between all convolutions.
Spatial Dropout is used regularizer in the bottleneck .
1 x 1 bottleneck convolutions reduce the dimensionality and expands the dimensionality.
In between these convolutions, a regular , dilated or full convolution takes place.
'''
def bottleneck(inputData, output, internal_scale=4, asymmetric=0, dilated=0, downsample=False, dropout_rate=0.1):
	internal = output // internal_scale
	encodeData = inputData
	if downsample:
		input_stride = 2
	else:
		input_stride=1 
	encodeData = Conv2D(internal, (input_stride, input_stride),
					 strides=(input_stride, input_stride), use_bias=False)(encoder)
	encodeData = BatchNormalization(momentum=0.1)(encodeData)  
	encodeData = PReLU(shared_axes=[1, 2])(encodeData)
	if not asymmetric and not dilated:
		encodeData = Conv2D(internal, (3, 3), padding='same')(encodeData)
	elif asymmetric:
		encodeData = Conv2D(internal, (1, asymmetric), padding='same', use_bias=False)(encodeData)
		encodeData = Conv2D(internal, (asymmetric, 1), padding='same')(encodeData)
	else :
		encodeData = Conv2D(internal, (3, 3), dilation_rate=(dilated, dilated), padding='same')(encodeData)
	

	encodeData = BatchNormalization(momentum=0.1)(encodeData)  
	encodeData = PReLU(shared_axes=[1, 2])(encodeData)
	encodeData = Conv2D(output, (1, 1), use_bias=False)(encodeData)
	encodeData = BatchNormalization(momentum=0.1)(encodeData) 
	encodeData = SpatialDropout2D(dropout_rate)(encodeData)
	prevData = inputData
	if downsample:
		prevData = MaxPooling2D()(prevData)

		prevData = Permute((1, 3, 2))(prevData)
		pad_feature_maps = prevData - inputData.get_shape().as_list()[3]
		tb_pad = (0, 0)
		lr_pad = (0, pad_feature_maps)
		prevData = ZeroPadding2D(padding=(tb_pad, lr_pad))(other)
		prevData = Permute((1, 3, 2))(other)

	encodeData = add([encodeData, prevData])
	encodeData = PReLU(shared_axes=[1, 2])(encodeData)

	return encodeData

'''
The encoder has three stages each consisting  of 5 bottleneck blocks . 
'''
def encoder(inp, dropout_rate=0.01):
    Effnet = initial_block(inp)
    Effnet = BatchNormalization(momentum=0.1)(Effnet) 
    Effnet = PReLU(shared_axes=[1, 2])(Effnet)
    Effnet = bottleneck(Effnet, 64, downsample=True, dropout_rate=dropout_rate)  # bottleneck 1.0
    for _ in range(4):
        Effnet = bottleneck(Effnet, 64, dropout_rate=dropout_rate)  # bottleneck 1.i

    Effnet = bottleneck(Effnet, 128, downsample=True)  # bottleneck 2.0
    # bottleneck 2.x and 3.x
    for _ in range(2):
        Effnet = bottleneck(Effnet, 128)             # bottleneck 2.1
        Effnet = bottleneck(Effnet, 128, dilated=2)  # bottleneck 2.2
        Effnet = bottleneck(Effnet, 128, asymmetric=5)  # bottleneck 2.3
        Effnet = bottleneck(Effnet, 128, dilated=4)  # bottleneck 2.4
        Effnet = bottleneck(Effnet, 128)            # bottleneck 2.5
        Effnet = bottleneck(Effnet, 128, dilated=8)  # bottleneck 2.6
        Effnet = bottleneck(Effnet, 128, asymmetric=5)  # bottleneck 2.7
        Effnet = bottleneck(Effnet, 128, dilated=16)  # bottleneck 2.8

    return Effnet
