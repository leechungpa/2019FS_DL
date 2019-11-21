
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
tf.compat.v1.enable_eager_execution()

import sys
import os

import warnings

from tensorflow import keras
import numpy as np
from tensorflow.data import Dataset
from image import *
import tensorflow.image

import numpy as np
import json
import cv2
import time


def load_data(paths, train = True):
    '''
    objective: load image files
    param: paths to each image files
    return: image files of input image and ground-truth image
    '''
    for img_path in paths:
        gt_path = img_path.decode("utf-8").replace('.jpg','.h5').replace('images','ground-truth')
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels = 3)
        img = tf.cast(img, tf.float32)
        img = (img/127.5) - 1 # normalizing the images to [-1, 1]

        gt_file = h5py.File(gt_path, 'r')
        target = np.asarray(gt_file['density'])

        target = cv2.resize(target,(int(target.shape[1]/8),int(target.shape[0]/8)),interpolation = cv2.INTER_CUBIC)*64


        yield (img, target)


def load_datasets():

	train_list, test_list = load_data_list()

	# load dataset from generator defined as load_data
	train_dataset = tf.data.Dataset.from_generator(
		load_data, args = [train_list],output_types = (tf.float32, tf.float32), output_shapes = ((None,None,3), (None,None)))
	train_dataset = train_dataset.shuffle(100000)

	test_dataset = tf.data.Dataset.from_generator(
		load_data, args = [test_list], output_types = (tf.float32, tf.float32), output_shapes = ((None,None,3), (None,None)))
	return train_dataset, test_dataset

def loss_fn(model, input_image, gt_image):
	'''
	objective: calculate loss from input image and ground-truth
	return: loss 
	'''
	output = model(np.expand_dims(input_image,0), training=True)
	output = tf.squeeze(output, [0,3])
    # mean squared error
	loss = tf.reduce_mean(tf.square(output - gt_image))
	return loss

def grad(model, input_image, gt_image):
	'''
	objective: apply gradient descent method to update model's weights
	'''
	with tf.GradientTape() as tape:
		loss = loss_fn(model, input_image, gt_image)
	return tape.gradient(loss, model.trainable_weights)

def fit(model, epochs, learning_rate = 0.01):

	train_dataset, test_dataset = load_datasets()
	optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
	
	# train model
	print('Learning started. It takes sometime.')

	for epoch in range(epochs):
		# init values
		avg_loss = 0.
		train_step = 0
		test_step = 0
		test_mae = 0
    
		# train process
		for step, (images, gt_images) in enumerate(train_dataset):

			grads = grad(model, images, gt_images)
			optimizer.apply_gradients(zip(grads, model.trainable_variables))
			loss = loss_fn(model, images, gt_images)
			avg_loss += loss
			train_step += 1

		avg_loss = avg_loss / train_step

		# test process
		for step, (images, gt_images) in enumerate(test_dataset): 

			output = model(np.expand_dims(images,0))
			test_step += 1

			test_mae += abs(np.sum(output)-np.sum(gt_images))

		test_mae = test_mae / test_step


		print('Epoch:', '{}'.format(epoch + 1), 'loss =', '{:.8f}'.format(avg_loss), 
		      'Test MAE = ', '{:.4f}'.format(test_mae))

		print('Learning Finished!')

