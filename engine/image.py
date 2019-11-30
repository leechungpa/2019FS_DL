import random
import os
import numpy as np
import tensorflow as tf
import h5py
import cv2

from IPython.display import HTML, display


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



# Custom IPython progress bar for training
class ProgressMonitor(object):
    
    tmpl = """
        <table style="width: 100%;">
            <tbody>
                <tr>
                    <td style="width: 30%;">
                     <b>Loss: {loss:0.4f}</b> &nbsp&nbsp&nbsp {value} / {length}
                    </td>
                    <td style="width: 70%;">
                        <progress value='{value}' max='{length}', style='width: 100%'>{value}</progress>
                    </td>
                </tr>
            </tbody>
        </table>        
        """

    def __init__(self, length):
        self.length = length
        self.count = 0
        self.display = display(self.html(0, 0), display_id=True)
        
    def html(self, count, loss):
        return HTML(self.tmpl.format(length=self.length, value=count, loss=loss))
        
    def update(self, count, loss):
        self.count += count
        self.display.update(self.html(self.count, loss))
