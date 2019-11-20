import random
import os
from PIL import Image,ImageFilter,ImageDraw
import numpy as np
import tensorflow as tf
import h5py
from PIL import ImageStat
import cv2


def load_train_data(train = True):
    for i in train_list:
        img_path = train_list[i]
        gt_path = img_path.replace('.jpg','.h5').replace('images','ground-truth')
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels = 3)
        img = tf.image.convert_image_dtype(img, tf.float32)

        gt_file = h5py.File(gt_path)
        target = np.asarray(gt_file['density'])

        target = cv2.resize(target,(int(target.shape[1]/8),int(target.shape[0]/8)),interpolation = cv2.INTER_CUBIC)*64

    
        yield (img, target)
        
def load_test_data(train = True):
    for i in test_list:
        img_path = test_list[i]
        gt_path = img_path.replace('.jpg','.h5').replace('images','ground-truth')
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels = 3)
        img = tf.image.convert_image_dtype(img, tf.float32)

        gt_file = h5py.File(gt_path)
        target = np.asarray(gt_file['density'])

        target = cv2.resize(target,(int(target.shape[1]/8),int(target.shape[0]/8)),interpolation = cv2.INTER_CUBIC)*64

    
        yield (img, target)
        
def load_data(img_path,train = True):
    gt_path = img_path.replace('.jpg','.h5').replace('images','ground-truth')
    #img = Image.open(img_path).convert('RGB')
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels = 3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    
    gt_file = h5py.File(gt_path)
    target = np.asarray(gt_file['density'])
    if False:
        crop_size = (img.size[0]/2,img.size[1]/2)
        if random.randint(0,9)<= -1:
            
            
            dx = int(random.randint(0,1)*img.size[0]*1./2)
            dy = int(random.randint(0,1)*img.size[1]*1./2)
        else:
            dx = int(random.random()*img.size[0]*1./2)
            dy = int(random.random()*img.size[1]*1./2)
        
        
        
        img = img.crop((dx,dy,crop_size[0]+dx,crop_size[1]+dy))
        target = target[dy:crop_size[1]+dy,dx:crop_size[0]+dx]
        
        
        
        
        if random.random()>0.8:
            target = np.fliplr(target)
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
    
    
    
    
    target = cv2.resize(target,(int(target.shape[1]/8),int(target.shape[0]/8)),interpolation = cv2.INTER_CUBIC)*64
    
    
    return img,target