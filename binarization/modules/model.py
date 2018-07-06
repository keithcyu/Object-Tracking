import os
import numpy as np
import scipy.io

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization

from layers import *
from metrics import *

sys.path.insert(0, '../..')
from binarization_utils import *

batch_norm_eps=1e-4
batch_norm_alpha=0.1#(this is same as momentum)
resid_levels = 4

def get_model(model_path=None, num_branches=1):
	model = Sequential()
	
	# model.add(Conv2D(96, kernel_size=(7,7), strides=(2,2), activation='relu', input_shape=(107,107, 3), name='conv1'))
	model.add(binary_conv(nfilters=96, ch_in=3, k=7, strides=(2,2), padding='valid', input_shape=[107,107,3]))
	# model.add(LRN())
	model.add(BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps))
	model.add(Residual_sign(levels=resid_levels))
	model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
	
	# model.add(Conv2D(256, kernel_size=(5,5), strides=(2,2), activation='relu', name='conv2'))
	model.add(binary_conv(nfilters=256, ch_in=96, k=5, strides=(2,2), padding='valid'))
	# model.add(LRN())
	model.add(BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps))
	model.add(Residual_sign(levels=resid_levels))
	model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))

	# model.add(Conv2D(512, kernel_size=(3,3), strides=(1,1), activation='relu', name='conv3'))
	model.add(binary_conv(nfilters=512, ch_in=256, k=3, strides=(1,1), padding='valid'))
	model.add(BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps))
	model.add(Residual_sign(levels=resid_levels))

	# model.add(Flatten())
	model.add(my_flat())

	model.add(Dropout(0.5))
	# model.add(Dense(512, activation='relu', name='fc4'))
	model.add(binary_dense(n_in=512*3*3, n_out=512))
	model.add(BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps))
	model.add(Residual_sign(levels=resid_levels))

	model.add(Dropout(0.5))
	# model.add(Dense(512, activation='relu', name='fc5'))
	model.add(binary_dense(n_in=512, n_out=512))
	model.add(BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps))
	model.add(Residual_sign(levels=resid_levels))

	model.add(Dropout(0.5)) # should be in each branch layer, but don't see why
	model.add(Branches(num_branches=num_branches, units=2))

	# load model
	if model_path is not None:
		if os.path.splitext(model_path)[1] == '.pth':
			load_model(model_path)
		elif os.path.splitext(model_path)[1] == '.mat':
			# load_mat_model(model, model_path)
			pass
		else:
			raise RuntimeError("Unknown model format: %s" % (model_path))

	return model

def load_mat_model(model, matfile):
	mat = scipy.io.loadmat(matfile)
	mat_layers = list(mat['layers'])[0]

	# copy conv weights
	for i, j in enumerate([0, 3, 6]):
		weight, bias = mat_layers[i*4]['weights'].item()[0]
		model.layers[j].set_weights([weight, bias[:,0]])
		#model.layers[i].set_weights([np.transpose(weight, (3,2,0,1)), bias[:,0]])
