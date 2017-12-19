#!/usr/bin/env python

import sys, nltk, pickle
from nltk.tokenize import RegexpTokenizer

data = pickle.load(sys.stdin.buffer) #work
tokenize = lambda raw: [token.casefold() for token in RegexpTokenizer(r'\w+').tokenize(raw)]
for doc_id in data.keys():
	tl_title = tokenize(data[doc_id][0]) #word set and token list
	tl_text = tokenize(data[doc_id][1])
	wordset = set(tl_text).union(set(tl_title))
	for term in wordset:
		print( "%s\t%s" % (term, doc_id) )