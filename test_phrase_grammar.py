import argparse
import re

import matplotlib.pyplot as plt
import nltk
import numpy
import tensorflow as tf
import tensorflow.nn as nn
# from torch.autograd import Variable

import data
import data_ptb
from utils import batchify, get_batch, repackage_hidden, evalb

from parse_comparison import corpus_stats_labeled, corpus_average_depth
from data_ptb import word_tags