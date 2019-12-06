# Data preprocessing file for the Penn-Treebank dataset
import os
import re
import pickle
import copy
from collections import defaultdict

import numpy as np
import tensorflow as tf
import nltk
from nltk.corpus import treebank as ptb

word_tags = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT',
             'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',
             'WDT', 'WP', 'WP$', 'WRB']

currency_tags_words = ['#', '$', 'C$', 'A$']
ellipsis = ['*', '*?*', '0', '*T*', '*ICH*', '*U*', '*RNR*', '*EXP*', '*PPA*', '*NOT*']
punctuation_tags = ['.', ',', ':', '-LRB-', '-RRB-', '\'\'', '``']
punctuation_words = ['.', ',', ':', '-LRB-', '-RRB-', '\'\'', '``', '--', ';', '-', '?', '!', '...', '-LCB-', '-RCB-']

file_ids = ptb.fileids()
train_file_ids = []
valid_file_ids = []
test_file_ids = []
rest_file_ids = []
for id in file_ids:
    if 'wsj_0001.mrg' <= id <= 'wsj_0199.MRG':
        train_file_ids.append(id)
    if 'wsj_0180.mrg' <= id <= 'wsj_0199.mrg':
        valid_file_ids.append(id)
    if 'wsj_0160.mrg' <= id <= 'wsj_0179.mrg':
        test_file_ids.append(id)

class Dictionary(object):
    def __init__(self):
        self.word2idx = {'<unk>': 0}
        self.idx2word = ['<unk>']
        self.word2frq = defaultdict(int)

    def __len__(self):
        return len(self.idx2word)
    
    def __getitem__(self, item):
        if item in self.word2idx:
            return self.word2idx[item]
        else:
            return self.word2idx['<unk>']
    
    def add(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        self.word2frq[word] += 1
        return self.word2idx[word]

    def rebuild_by_freq(self, thd=3):
        self.word2idx = {'<unk>': 0}
        self.idx2word = ['<unk>']

        for k, v in self.word2frq.items():
            if v >= thd and k not in self.idx2word:
                self.idx2word.append(k)
                self.word2idx[k] = len(self.idx2word) - 1

class Corpus(object):
    def __init__(self, pathToData):
        dict_file_name = os.path.join(pathToData, 'dict.pkl')
        self.dict = Dictionary()
        self.add_words(train_file_ids)
        self.dict.rebuild_by_freq()

        pickle.dump(self.dict, open(dict_file_name, 'wb'))

        self.train, self.train_sens, self.train_trees, self.train_nltktrees = self.tokenize(train_file_ids)
        self.valid, self.valid_sens, self.valid_trees, self.valid_nltktress = self.tokenize(valid_file_ids)
        self.test, self.test_sens, self.test_trees, self.test_nltktrees = self.tokenize(test_file_ids)
        self.rest, self.rest_sens, self.rest_trees, self.rest_nltktrees = self.tokenize(rest_file_ids)
    
    @staticmethod
    def _filter_words(tree):
        words = []
        for w, tag in tree.pos():
            if tag in word_tags:
                w = w.lower()
                w = re.sub('[0-9]+', 'N', w)
                words.append(w)
        return words

    def add_words(self, file_ids):
        for id in file_ids:
            sentences = ptb.parsed_sents(id)
            for sen_tree in sentences:
                words = Corpus._filter_words(sen_tree)
                words = ['<eos>'] + words + ['<eos>']
                for word in words:
                    self.dict.add(word)
    
    def tokenize(self, file_ids):
        def tree2list(tree):
            if not isinstance(tree, nltk.Tree):
                return []
            elif tree.label() in word_tags:
                w = tree.leaves()[0].lower()
                w = re.sub('[0-9]+', 'N', w)
                return w

            root = []
            for child in tree:
                c = tree2list(child)
                if c is not None:
                    root.append(c)

            if len(root) > 1:
                return root
            elif len(root) == 1:
                return root[0]
        
        sens_idx = []
        sens = []
        trees = []
        nltk_trees = []

        for id in file_ids:
            sentences = ptb.parsed_sents(id)
            for sen_tree in sentences:
                words = self._filter_words(sen_tree)
                words = ['<eos>'] + words + ['<eos>']

                sens.append(words)
                idx = []

                for word in words:
                    idx.append(self.dict[word])

                sens_idx.append(idx)
                trees.append(tree2list(sen_tree))
                nltk_trees.append(sen_tree)

        return sens_idx, sens, trees, nltk_trees

corp = Corpus(os.getcwd())
print(corp.train_nltktrees[0].pos())