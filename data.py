# Data preprocessing file for the Penn-Treebank dataset
import os
import re
import pickle
import copy
from collections import defaultdict

import numpy as np
import tensorflow as tf

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.word2frq = defaultdict(int)

    def __len__(self):
        return len(self.idx2word)
    
    def __getitem__(self, item):
        if item in self.word2idx:
            return self.word2idx[item]
        else:
            return None
    
    def add(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        self.word2frq[word] += 1
        return self.word2idx[word]

class Corpus(object):
    def __init__(self, pathToData):
        self.dict = Dictionary()
        self.train = self.tokenize(pathToData + "train.txt")
        self.val = self.tokenize(pathToData + "valid.txt")
        self.test = self.tokenize(pathToData + "test.txt")
    
    def tokenize(self, filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError
        
        with open(filepath, 'r') as f:
            data = f.read().replace('\n', '')

        data = data.split('.')
        ids = []
        for sentence in data:
            sentence = sentence.split() + ['<eos>']
            for word in sentence:
                self.dict.add(word)
                ids.append(self.dict.word2idx[word])
        
        return ids


pathToData = os.getcwd() + "/data/penn/"
corpus = Corpus(pathToData)
print(corpus.train[-5:])