#!/usr/bin/env python3    
# -*- coding: utf-8 -*- 

from gensim.models import word2vec
from os.path import join, exists, split
import os



def train_word2vec(ques_pairs, num_features, min_word_count, context):
    model_dir = '../data/word2vec_models'
    model_name = "word2vec.mdl"
    model_name = join(model_dir, model_name)
    

    sentences = []
    for pair in ques_pairs:
        sentences.append(pair[0])
        sentences.append(pair[1])
        
    embedding_model = word2vec.Word2Vec(sentences, 
                        size=num_features, min_count=min_word_count,
                        workers=2, \
                        window=context)
    
    embedding_model.init_sims(replace=True)
    
    # Saving the model for later use. You can load it later using Word2Vec.load()
    if not exists(model_dir):
        os.mkdir(model_dir)
    print ('Saving Word2Vec model')
    embedding_model.save(model_name)
    
    return embedding_model