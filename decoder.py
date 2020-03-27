from keras.layers.core import Activation,SpatialDropout2D, Permute,  Reshape
from keras.layers.merge import add, concatenate
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.engine.topology import Input
from keras.models import Model


def decodeNeck(encoder, output, upsample=False, reverse_module=False):
	internal = output // 4

	x = Conv2D(internal, (1, 1), use_bias=False)(encoder)
	x = BatchNormalization(momentum=0.1)(x)
	x = Activation('relu')(x)
	if not upsample:
		x = Conv2D(internal, (3, 3), padding='same', use_bias=True)(x)
	else:
		x = Conv2DTranspose(filters=internal, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
	x = BatchNormalization(momentum=0.1)(x)
	x = Activation('relu')(x)

	x = Conv2D(output, (1, 1), padding='same', use_bias=False)(x)

	other = encoder
	if encoder.get_shape()[-1] != output or upsample:
		other = Conv2D(output, (1, 1), padding='same', use_bias=False)(other)
		other = BatchNormalization(momentum=0.1)(other)
		if upsample and reverse_module is not False:
			other = UpSampling2D(size=(2, 2))(other)

	if upsample and reverse_module is False:
		decoder = x
	else:
		x = BatchNormalization(momentum=0.1)(x)
		decoder = add([x, other])
		decoder = Activation('relu')(decoder)

	return decoder
'''
The decoder with Stage 4 has 3 bottlenecks and Stage 5 has 2 bottlenecks
'''
def decoder(encoder, nc):
	Effnet = decodeNeck(encoder, 64, upsample=True, reverse_module=True)  # bottleneck 4.0
	Effnet = decodeNeck(Effnet, 64)  # bottleneck 4.1
	Effnet = decodeNeck(Effnet, 64)  # bottleneck 4.2
	Effnet = decodeNeck(Effnet, 16, upsample=True, reverse_module=True)  # bottleneck 5.0
	Effnet = decodeNeck(Effnet, 16)  # bottleneck 5.1

	Effnet = Conv2DTranspose(filters=nc, kernel_size=(2, 2), strides=(2, 2), padding='same')(Effnet)
	return Effnet
