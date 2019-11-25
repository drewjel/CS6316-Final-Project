'''
Created on Oct 28, 2019

@author: andrewelsey
'''

import numpy as np
import tensorflow as tf
import yaml


from ordered_neuron_model import OrderedNeuronModel

from data import Corpus


model = 'train'

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data[0: nbatch * bsz]
    # Evenly divide the data across the bsz batches.
    #data = data.view(bsz, -1).t().contiguous()
    data = data.reshape((bsz, -1)).transpose()
    return data


def get_batch(source, i, seq_len):
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape((-1))
    return data, target


if __name__ == '__main__':
    import os

    c = os.getcwd()
    #cfg = yaml.load('/CS6316-Final-Project/run_config.yaml')
    with open("CS6316-Final-Project\\run_config.yaml", 'r') as strm:
        cfg = yaml.safe_load(strm)

    print(cfg)
    
    vocab_size = None
    chunk_size_factor = cfg['model']['chunk_size_factor']
    hidden_size = cfg['model']['hidden_size']
    nlayers = cfg['model']['nlayers']
    batch_size = cfg['model']['batch_size']
    input_size = cfg['model']['input_size']
    
    epochs = cfg['model']['epochs']

    lr = float(cfg['model']['lr'])

    seq_len = cfg['model']['seq_len']
    #(self, C, nlayers, vocab_size, input_size, hidden_size, lr)

    corpus_loc = cfg['corpus_loc']
    print('test')
    print(corpus_loc)
    corpus = Corpus(corpus_loc)
    
    vocab_size = len(corpus.dict)


    train_batch_size = cfg['model']['train_batch_size']
    eval_batch_size = cfg['model']['eval_batch_size']
    test_batch_size = cfg['model']['test_batch_size']

    corpus_train = batchify(corpus.train, train_batch_size)
    corpus_valid = batchify(corpus.val, eval_batch_size)
    corpus_test = batchify(corpus.test, test_batch_size)
    
    batch_no = 0%corpus_train.shape[0]

    batch = get_batch(corpus_train, batch_no, train_batch_size)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        with tf.device("/GPU:0"):
            model = OrderedNeuronModel(chunk_size_factor, nlayers, vocab_size, input_size, hidden_size, lr, train_batch_size)
            init = tf.initialize_all_variables()
            sess.run(init)
        losses = np.zeros([200])
        for i in range(epochs):

            batch_no = i%corpus_train.shape[0]

            batch = get_batch(corpus_train, batch_no, train_batch_size)

            for j in batch[1]:
                if isinstance(j, str):
                    print('wtf')

            for j in batch[0].reshape((-1)):
                if isinstance(j, str):
                    print('wtf')

            #print('Running epoch ' + str(i))
            
            _, losses[i%200] = sess.run([model.step, model.loss], feed_dict={model.cell.input:batch[0], model.cell.seq_len:seq_len, model.targets:batch[1].reshape((-1, 1))})

            #print(losses[i%100])

            if i>= 200 and i%200 == 0:
                print('AVERAGE')
                print(np.average(losses))

            #_ = sess.run([model.step], feed_dict={model.cell.input:batch[0], model.cell.seq_len:seq_len, model.targets:batch[1].reshape((-1, 1))})
