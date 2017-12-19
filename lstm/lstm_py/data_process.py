#!/usr/bin/env python3    
# -*- coding: utf-8 -*- 

import numpy as np
from keras.preprocessing.sequence import pad_sequences
import collections
import csv
from nltk.tokenize import word_tokenize
from os.path import join, exists
import os
import pickle


def parse_quora_dul_data(in_file, num):
    out_dir = "../data/preprocessed"
    out_name = "preprocessed.pkl"
    out_name = join(out_dir, out_name)
    if not exists(out_dir):
        os.mkdir(out_dir)
    ques_pairs = []
    with open(in_file, newline = '') as fi:
        infilereader = csv.reader(fi)
        next(infilereader, None)
        for line in infilereader:
            if len(line) == 0:
                continue
            pid = line[0]
            ques1 = line[3]  
            ques2 = line[4] 
            is_dul = int(line[5])  
            ques1 = ques1.strip('"')
            ques2 = ques2.strip('"')
            ques1_token = word_tokenize(ques1)
            ques2_token = word_tokenize(ques2) 
            
            ques_pairs.append((ques1_token, ques2_token, is_dul, pid))
            if len(ques_pairs) > num:
                break
        pickle.dump(ques_pairs, open(out_name, 'wb'))
             
    return ques_pairs


def build_index(ques_pairs):
    out_dir = "../data/word2index"
    out_name = "word2index.pkl"
    out_name = join(out_dir, out_name)

    if not exists(out_dir):
        os.mkdir(out_dir)
    wordcounts = collections.Counter()

    for pair in ques_pairs:
        for w in pair[0]:
            wordcounts[w] += 1
        for w in pair[1]:
            wordcounts[w] += 1
            
    words = [wordcount[0] for wordcount in wordcounts.most_common()]
    word2index = {w: i + 1 for i, w in enumerate(words)}  # 0 = mask

    pickle.dump(word2index, open(out_name, 'wb'))
    return word2index


def get_seq_maxlen(ques_pairs):
    out_dir = "../data/seq_maxlen"
    out_name = "seq_maxlen.pkl"
    out_name = join(out_dir, out_name)

    if not exists(out_dir):
        os.mkdir(out_dir)
    max_ques1_len = max([len(pair[0]) for pair in ques_pairs])
    max_ques2_len = max([len(pair[1]) for pair in ques_pairs])
    max_seq_len = max([max_ques1_len, max_ques2_len])
    pickle.dump(max_seq_len, open(out_name, 'wb'))
    return max_seq_len

    
def vectorize_ques_pair(ques_pairs, word2idx, seq_maxlen): 
    x_ques1 = []
    x_ques2 = []
    y = []
    pids = []
    for pair in ques_pairs:
        x_ques1.append([word2idx[w] for w in pair[0]])
        x_ques2.append([word2idx[w] for w in pair[1]])
        y.append((np.array([0, 1]) if pair[2] == 1 else np.array([1, 0])))
        pids.append(pair[3])
                 
    x_ques1 = pad_sequences(x_ques1, maxlen=seq_maxlen)
    x_ques2 = pad_sequences(x_ques2, maxlen=seq_maxlen)
    y = np.array(y)
    pids = np.array(pids)
    
    return x_ques1, x_ques2, y, pids
