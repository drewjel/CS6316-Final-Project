'''
Created on Oct 28, 2019

@author: andrewelsey
'''

import numpy as np
import tensorflow as tf
import yaml


from ordered_neuron_model import OrderedNeuronModel

from data import *


model = 'train'

if __name__ == '__main__':
    cfg = yaml.load('run_config.yaml')
    
    vocab_size = None
    chunk_size_factor = cfg['model']['chunk_size_factor']
    hidden_size = cfg['model']['hidden_size']
    nlayers = cfg['model']['nlayers']
    batch_size = cfg['model']['batch_size']
    input_size = cfg['model']['input_size']
    
    model = OrderedNeuronModel(chunk_size_factor, nlayers, input_size, hidden_size)
    
    with tf.Session() as sess:
        tf.initialize_all_variables()