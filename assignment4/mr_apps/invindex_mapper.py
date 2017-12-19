#!/usr/bin/env python


import sys, nltk, pickle
from nltk.tokenize import RegexpTokenizer


tokenize = lambda raw: [token.casefold() for token in RegexpTokenizer(r'\w+').tokenize(raw)]

'''def tokenize2(raw):
    tokenizer = RegexpTokenizer(r'\w+')
    token_list = tokenizer.tokenize(raw)
    tl = [token.casefold() for token in token_list]
    return  tl'''
def getTF(  tl_title,  tl_text) : # for one page
    tf = {}
    ws_text = set(tl_text)
    for word in ws_text: #word in the text set no repetitous word
        tf[word] = tl_text.count(word) 

    ws_title = set(tl_title)
    for word in  ws_title:
        score =  tl_title.count(word)
        if (not tf.get(word)): 
            tf[word] = 0
        tf[word] += score
    return tf

data = pickle.load(sys.stdin.buffer) #work
for doc_id in data.keys():
    tl_title = tokenize(data[doc_id][0]) #word set and token list
    tl_text = tokenize(data[doc_id][1])
    tf  = getTF(  tl_title,  tl_text)
    
    for term in tf.keys():
        print( '%s\t%s\t%d' % (doc_id, term, tf[term]) )  
