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
    train_a_list = []
    with open('engine/train_a_list.txt', 'r') as f:
        for line in f:
            inner_list = [elt.strip() for elt in line.split(',')]
            train_a_list.append(inner_list)
    train_a_list = [val.replace('\'','') for val in train_a_list[0]]

    test_a_list = []
    with open('engine/test_a_list.txt', 'r') as f:
        for line in f:
            inner_list = [elt.strip() for elt in line.split(',')]
            test_a_list.append(inner_list)
    test_a_list = [val.replace('\'','') for val in test_a_list[0]]

    train_b_list = []
    with open('engine/train_b_list.txt', 'r') as f:
        for line in f:
            inner_list = [elt.strip() for elt in line.split(',')]
            train_b_list.append(inner_list)
    train_b_list = [val.replace('\'','') for val in train_b_list[0]]

    test_b_list = []
    with open('engine/test_b_list.txt', 'r') as f:
        for line in f:
            inner_list = [elt.strip() for elt in line.split(',')]
            test_b_list.append(inner_list)
    test_b_list = [val.replace('\'','') for val in test_b_list[0]]


    return train_a_list, test_a_list, train_b_list, test_b_list


