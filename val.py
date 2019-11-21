import tensorflow as tf
import h5py
import numpy as np
import cv2
from matplotlib import cm as c
from matplotlib import pyplot as plt

def generate_images(model, test_input = 'ShanghaiTech/part_A/test_data/images/IMG_100.jpg'):
    img = tf.io.read_file(test_input)
    img = tf.image.decode_jpeg(img, channels = 3)
    img = tf.cast(img, tf.float32)
    img = (img/127.5) - 1 
    
    gt_path = test_input.replace('.jpg','.h5').replace('images','ground-truth')
    gt_file = h5py.File(gt_path, 'r')
    target = np.asarray(gt_file['density'])
    target = cv2.resize(target,(int(target.shape[1]/8),int(target.shape[0]/8)),interpolation = cv2.INTER_CUBIC)*64

    prediction = model(np.expand_dims(img,0), training=True)
    prediction = np.asarray(tf.squeeze(prediction, [0,3]))
    
    print("Predicted Count : ",int(np.sum(prediction)))
    plt.imshow(prediction, cmap = c.jet)
    plt.show()
    print("Original Count : ",int(np.sum(target)) + 1)
    plt.show()
    print("Original Image")
    plt.imshow(plt.imread(test_input))
    plt.show()