# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import toimage
from keras.datasets import cifar10

from keras.models import Sequential

from keras.models import Model

from keras.layers import (
    Input,
    Activation,
    Dense,
    Flatten,
    ZeroPadding2D,
    BatchNormalization,
)

from keras.engine.topology import Merge

from keras.layers.convolutional import (
    Conv2D,
    MaxPooling2D,
    AveragePooling2D
)


from keras.callbacks import TensorBoard


from keras.datasets import cifar10
from keras.utils import np_utils

import matplotlib.pyplot as plt
#from keras.utils.visualize_util import plot
#from keras.utils.vis_utils import plot_model


class ResnetModel(object):
	
	def __init__(self, shape, classes, blocks=[2,2,2,2]):


		self.axis = 3

		inputs = Input(shape=shape)

		#x = Conv2D(64, 7,7, border_mode='same', subsample=(2,2), init='he_normal')(inputs)
		x = Conv2D(64, 3,3, border_mode='same', subsample=(1,1), init='he_normal', bias=False)(inputs)
		x = BatchNormalization(axis=self.axis)(x)
		x = Activation('relu')(x)
		#x = MaxPooling2D((3,3), border_mode='same', strides=(2,2))(x)

		channel_size = int(x.shape[self.axis]) # x.shape[index] : Dimension type

		print(channel_size)
		for stage, block_size in enumerate(blocks):
			for j in range(block_size):
				x = self.block(x, channel_size)

			channel_size *= 2

		print(x.shape)
		print(classes)
		x = Flatten()(x)
		predictions = Dense(classes, activation='softmax')(x)
		
		self.model = Model(inputs, predictions)
		self.model.compile("sgd", "categorical_crossentropy", ["accuracy"])

		#plot(model, to_file='./model.png')


	def block(self, inputs, channel_size ):
		
		change_size = inputs.shape[self.axis] != channel_size
		print(change_size)

		if change_size:
			strides = (2,2)
		else:
			strides = (1,1)

		#x = Conv2D(64, 3,3, padding='same')(inputs)
		x = Conv2D(channel_size, 3,3, border_mode='same', init='he_normal', subsample=strides, bias=False)(inputs)
		x = BatchNormalization(axis=self.axis)(x)
		x = Activation('relu')(x)
		x = Conv2D(channel_size, 3,3, border_mode='same', init='he_normal', bias=False)(x)
		x = BatchNormalization(axis=self.axis)(x)

		if change_size:
			shortcut = Conv2D(int(x.shape[self.axis]), 1,1, border_mode='same', init='he_normal', subsample=strides, bias=False)(inputs)
			shortcut = BatchNormalization(axis=self.axis)(shortcut)
		else:
			shortcut = inputs

		x = Merge(mode='sum')([x, shortcut])
		x = Activation('relu')(x)
		return x

	

	def learn(self, x_train, y_train,  x_test, y_test, batch_size=128, nb_epoch=30):
		hist = self.model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_split=0.01)

		score=self.model.evaluate(x_test,y_test,verbose=1)
		print('Test loss:',score[0])
		print('Test accuracy:',score[1])

		loss = hist.history['loss']
		val_loss = hist.history['val_loss']

		# lossのグラフ
		plt.plot(range(nb_epoch), loss, marker='.', label='loss')
		plt.plot(range(nb_epoch), val_loss, marker='.', label='val_loss')
		plt.legend(loc='best', fontsize=10)
		plt.grid()
		plt.xlabel('epoch')
		plt.ylabel('loss')
		plt.show()

if __name__ == '__main__':
	shape, classes = (32, 32, 3), 10

	# CIFAR-10データセットをロード
	(X_train, y_train), (X_test, y_test) = cifar10.load_data()
	print(" ------------------------ ")
	print(" Train Data : ")
	print(X_train.shape, y_train.shape)
	print(" Test Data : ")
	print(X_test.shape, y_test.shape)
	print(" ------------------------ ")


	# floatに型変換
	X_train = X_train.astype('float32')
	X_test = X_test.astype('float32')
	# 各画素値を正規化
	X_train /= 255.0
	X_test /= 255.0

	# one hot
	Y_train = np_utils.to_categorical(y_train, classes)
	Y_test = np_utils.to_categorical(y_test, classes)


	model = ResnetModel( shape, classes)

	model.learn( X_train, Y_train, X_test, Y_test)

