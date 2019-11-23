import random
import os
import numpy as np
import tensorflow as tf
import h5py
import cv2

def load_data_list():
    '''
    objective: load lists of data files
    return: lists
    '''
    train_list = []
    with open('engine/train_data_list.txt', 'r') as f:
        for line in f:
            inner_list = [elt.strip() for elt in line.split(',')]
            train_list.append(inner_list)
    train_list = [val.replace('\'','') for val in train_list[0]]

    test_list = []
    with open('engine/test_data_list.txt', 'r') as f:
        for line in f:
            inner_list = [elt.strip() for elt in line.split(',')]
            test_list.append(inner_list)
    test_list = [val.replace('\'','') for val in test_list[0]]

    return train_list, test_list


