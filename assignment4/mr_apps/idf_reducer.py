#!/usr/bin/env python

from itertools import groupby
from operator import itemgetter
import sys,pickle
from math import log10
# from assignment4.reformatter import PageNode
idf = lambda q_df, doc_num: log10(doc_num/q_df)
result = {}
data = map(lambda x: x.strip().split('\t'), sys.stdin)

data =  sorted(data,key= itemgetter(0))
page_num = 0
for term, id in data:
	page_num+=1
for k_group, group in groupby(data, itemgetter(0) ): #group by term (doc_id, (word, TITLE_WORD_SCORE))
	# total = sum(int(id) for term,id in group)
	# print("%s" % (k_group))
	# result[k_group] = len( list(group) )
	result[k_group] = idf(  len( list(group) ), page_num) 
pickle.dump(result , sys.stdout.buffer)  
