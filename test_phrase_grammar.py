import argparse
import re

import matplotlib.pyplot as plt
import nltk
nltk.download('treebank')
import numpy
import tensorflow as tf
import tensorflow.nn as nn

import data
import data_ptb
from utils import batchify, get_batch, repackage_hidden, evalb

from parse_comparison import corpus_stats_labeled, corpus_average_depth
from data_ptb import word_tags

# Test model
def build_tree(depth, sen):
    assert len(depth) == len(sen)

    if len(depth) == 1:
        parse_tree = sen[0]
    else:
        idx_max = numpy.argmax(depth)
        parse_tree = []
        if len(sen[:idx_max]) > 0:
            tree0 = build_tree(depth[:idx_max], sen[:idx_max])
            parse_tree.append(tree0)
        tree1 = sen[idx_max]
        if len(sen[idx_max + 1:]) > 0:
            tree2 = build_tree(depth[idx_max + 1:], sen[idx_max + 1:])
            tree1 = [tree1, tree2]
        if parse_tree == []:
            parse_tree = tree1
        else:
            parse_tree.append(tree1)
    return parse_tree

def get_brackets(tree, idx=0):
    brackets = set()
    if isinstance(tree, list) or isinstance(tree, nltk.Tree):
        for node in tree:
            node_brac, next_idx = get_brackets(node, idx)
            if next_idx - idx > 1:
                brackets.add((idx, next_idx))
                brackets.update(node_brac)
            idx = next_idx
        return brackets, idx
    else:
        return brackets, idx + 1

def MRG(tr):
    if isinstance(tr, str):
        return tr + ' '
    else:
        s = '( '
        for subtr in tr:
            s += MRG(subtr)
        s += ') '
        return s

def MRG_labeled(tr):
    if isinstance(tr, nltk.Tree):
        if tr.label() in word_tags:
            return tr.leaves()[0] + ' '
        else:
            s = '(%s ' % (re.split(r'[-=]', tr.label())[0])
            for subtr in tr:
                s += MRG_labeled(subtr)
            s += ') '
            return s
    else:
        return ''

import tensorflow as tf

def test(model, corpus, sess, seq_len):
    prt = True
    corpus = data_ptb.Corpus('data/penn')

    prec_list = []
    reca_list = []
    f1_list = []

    pred_tree_list = []
    targ_tree_list = []

    nsens = 0
    word2idx = corpus.dict.word2idx
    if True:#args.wsj10:
        dataset = zip(corpus.train_sens, corpus.train_trees, corpus.train_nltktrees)
    else:
        dataset = zip(corpus.test_sens, corpus.test_trees, corpus.test_nltktrees)

    corpus_sys = {}
    corpus_ref = {}
    print(len(corpus.test_sens))
    for sen, sen_tree, sen_nltktree in dataset:
        if len(sen) > 12:#args.wsj10 and len(sen) > 12:
            continue
        input = numpy.array([word2idx[w] if w in word2idx else word2idx['<unk>'] for w in sen])
        #print(input.shape)
        input = numpy.stack([input] + [numpy.zeros(input.shape) for i in range(79)])

        #print(input.shape)

        _, _, distance_forget, distance_input =\
           sess.run([model.cell.forward_propagate(input.shape[1])], feed_dict={model.cell.input:input, model.cell.seq_len:seq_len, model.targets:numpy.zeros((80,1))})[0]

        #print(distance_forget.shape)
        #print(distance_input.shape)

        distance_forget = distance_forget[:,:,0]
        distance_input = distance_input[:,:,0]

        nsens += 1
        if prt and nsens % 100 == 0:
            for i in range(len(sen)):
                print('%15s\t%s\t%s' % (sen[i], str(distance_forget[:, i]), str(distance_input[:, i])))
            print('Standard output:', sen_tree)

        sen_cut = sen[1:-1]
        for gates in [
            # distance[0],
            distance_forget[1],
            # distance[2],
            # distance.mean(axis=0)
        ]:
            #print(gates.shape)
            #print(len(sen_cut))
            depth = gates[1:-1]
            parse_tree = build_tree(depth, sen_cut)

            corpus_sys[nsens] = MRG(parse_tree)
            corpus_ref[nsens] = MRG_labeled(sen_nltktree)

            pred_tree_list.append(parse_tree)
            targ_tree_list.append(sen_tree)

            model_out, _ = get_brackets(parse_tree)
            std_out, _ = get_brackets(sen_tree)
            overlap = model_out.intersection(std_out)

            prec = float(len(overlap)) / (len(model_out) + 1e-8)
            reca = float(len(overlap)) / (len(std_out) + 1e-8)
            if len(std_out) == 0:
                reca = 1.
                if len(model_out) == 0:
                    prec = 1.
            f1 = 2 * prec * reca / (prec + reca + 1e-8)
            prec_list.append(prec)
            reca_list.append(reca)
            f1_list.append(f1)

            if prt and nsens % 1 == 0:
                print('Model output:', parse_tree)
                print('Prec: %f, Reca: %f, F1: %f' % (prec, reca, f1))

        if prt and nsens % 100 == 0:
            print('-' * 80)

            _, axarr = plt.subplots(3, sharex=True, figsize=(distance_forget.shape[1] // 2, 6))
            axarr[0].bar(numpy.arange(distance_forget.shape[1])-0.2, distance_forget[0], width=0.4)
            axarr[0].bar(numpy.arange(distance_input.shape[1])+0.2, distance_input[0], width=0.4)
            axarr[0].set_ylim([0., 1.])
            axarr[0].set_ylabel('1st layer')
            axarr[1].bar(numpy.arange(distance_forget.shape[1]) - 0.2, distance_forget[1], width=0.4)
            axarr[1].bar(numpy.arange(distance_input.shape[1]) + 0.2, distance_input[1], width=0.4)
            axarr[1].set_ylim([0., 1.])
            axarr[1].set_ylabel('2nd layer')
            axarr[2].bar(numpy.arange(distance_forget.shape[1]) - 0.2, distance_forget[2], width=0.4)
            axarr[2].bar(numpy.arange(distance_input.shape[1]) + 0.2, distance_input[2], width=0.4)
            axarr[2].set_ylim([0., 1.])
            axarr[2].set_ylabel('3rd layer')
            plt.sca(axarr[2])
            plt.xlim(xmin=-0.5, xmax=distance_forget.shape[1] - 0.5)
            plt.xticks(numpy.arange(distance_forget.shape[1]), sen, fontsize=10, rotation=45)

            plt.savefig('figure/%d.png' % (nsens))
            plt.close()

    prec_list, reca_list, f1_list \
        = numpy.array(prec_list).reshape((-1,1)), numpy.array(reca_list).reshape((-1,1)), numpy.array(f1_list).reshape((-1,1))
    if prt:
        print('-' * 80)
        numpy.set_printoptions(precision=4)
        print('Mean Prec:', prec_list.mean(axis=0),
              ', Mean Reca:', reca_list.mean(axis=0),
              ', Mean F1:', f1_list.mean(axis=0))
        print('Number of sentence: %i' % nsens)

        correct, total = corpus_stats_labeled(corpus_sys, corpus_ref)
        print(correct)
        print(total)
        print('ADJP:', correct['ADJP'], total['ADJP'])
        print('NP:', correct['NP'], total['NP'])
        print('PP:', correct['PP'], total['PP'])
        print('INTJ:', correct['INTJ'], total['INTJ'])
        print(corpus_average_depth(corpus_sys))

        evalb(pred_tree_list, targ_tree_list)

    return f1_list.mean(axis=0)