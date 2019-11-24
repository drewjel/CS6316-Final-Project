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
        
        print(filepath)
        if not os.path.exists(filepath):
            raise FileNotFoundError

        with open(filepath, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dict.add(word)

        # Tokenize file content
        with open(filepath, 'r') as f:
            ids = np.zeros((tokens), dtype=np.int)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dict.word2idx[word]
                    token += 1

        return ids


#pathToData = os.getcwd() + "/data/penn/"
#corpus = Corpus(pathToData)
#print(corpus.train[-5:])