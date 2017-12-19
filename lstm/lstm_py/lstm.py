#!/usr/bin/env python3
# -*- coding: utf-8 -*- 
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Merge, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from sklearn.cross_validation import train_test_split
import numpy as np
import os
import time
import argparse

import data_process
import word2vec_pretrain

def model(in_file, num):
    start = time.time()
    print("data process...")
    ques_pairs = data_process.parse_quora_dul_data(in_file, num)
    word2index = data_process.build_index(ques_pairs)
    vocab_size = len(word2index) + 1
    seq_maxlen = data_process.get_seq_maxlen(ques_pairs)
    x_ques1, x_ques2, y, pids = data_process.vectorize_ques_pair(ques_pairs, word2index, seq_maxlen)
    print(x_ques1.shape)
    x_ques1train, x_ques1test, x_ques2train, x_ques2test, ytrain, ytest, pidstrain, pidstest = train_test_split(x_ques1, x_ques2, y, pids, test_size=0.2, random_state=42)
    
    
    w2v_embedding_model = word2vec_pretrain.train_word2vec(ques_pairs, num_features=64, min_word_count=1, context=5)
    embedding_weights = np.zeros((vocab_size, 64))
    for word, index in word2index.items():
        if word in w2v_embedding_model:
            embedding_weights[index, :] = w2v_embedding_model[word]
        else:
            embedding_weights[index, :] = np.random.uniform(-0.25, 0.25, 
                                                            w2v_embedding_model.vector_size)
    print("Word2Vec train costs:", time.time() - start)
    
    print("Building model...")
    ques1_enc = Sequential()
    ques1_enc.add(Embedding(output_dim=64, input_dim=vocab_size, weights=[embedding_weights], mask_zero=True))
    ques1_enc.add(LSTM(100, input_shape=(64, seq_maxlen), return_sequences=False))
    ques1_enc.add(Dropout(0.3))
    
    ques2_enc = Sequential()
    ques2_enc.add(Embedding(output_dim=64, input_dim=vocab_size, weights=[embedding_weights], mask_zero=True))
    ques2_enc.add(LSTM(100, input_shape=(64, seq_maxlen), return_sequences=False))
    ques2_enc.add(Dropout(0.3))
    
    model = Sequential()
    model.add(Merge([ques1_enc, ques2_enc], mode="sum"))
    model.add(Dense(2, activation="softmax"))
    
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    print("Building model costs:", time.time() - start)
    
    print("Training...")
    checkpoint = ModelCheckpoint(filepath=os.path.join("../data/", "quora_dul_best_lstm.hdf5"), verbose=1, save_best_only=True)
    model.fit([x_ques1train, x_ques2train], ytrain, batch_size=32, epochs=1, validation_split=0.1, verbose=2, callbacks=[checkpoint])
    print("Training neural network costs:", time.time() - start)
    
    # predict
    print ("predict...")
    y_test_pred = model.predict_classes([x_ques1test, x_ques2test], batch_size=32)
    
    print("Evaluation...")
    loss, acc = model.evaluate([x_ques1test, x_ques2test], ytest, batch_size=32)
    print("Test loss/accuracy final model = %.4f, %.4f" % (loss, acc))
    
    model.save_weights(os.path.join("../data/", "quora_dul_lstm-final.hdf5"))
    with open(os.path.join("../data/", "quora_dul_lstm.json"), "w") as fjson:
        fjson.write(model.to_json())
    
    model.load_weights(filepath=os.path.join("../data/", "quora_dul_best_lstm.hdf5"))
    loss, acc = model.evaluate([x_ques1test, x_ques2test], ytest, batch_size=32)
    print("Test loss/accuracy best model = %.4f, %.4f" % (loss, acc))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--size")
    args = parser.parse_args()
    size = int(args.size)
    model("../data/train.csv", size)