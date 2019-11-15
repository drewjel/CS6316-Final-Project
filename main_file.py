'''
Created on Oct 28, 2019

@author: andrewelsey
'''

import numpy as np
import tensorflow as tf
import yaml


import on_lstm_model

if __name__ == '__main__':
    cfg = yaml.load('run_config.yaml')
    
    
    
    with tf.Session() as sess:
        tf.initialize_all_variables()