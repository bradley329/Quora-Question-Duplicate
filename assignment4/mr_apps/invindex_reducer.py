#!/usr/bin/env python

from itertools import groupby
from operator import itemgetter
import sys,pickle


# data = map(lambda x: x.strip().split('\t',2), [line for line in sys.stdin.buffer])
# data is list of [[doc_id, term, tf]]
result = {}
data = map(lambda x: x.strip().split('\t'), sys.stdin)
#data =  sorted(data,key= itemgetter(1)) 
for k_group, group in groupby(sorted(data,key= itemgetter(1)), itemgetter(1) ): #group by term (doc_id, (word, TITLE_WORD_SCORE))
    # print('%s\t' % k_group, end ='')
    # print( [(doc_id, tf) for doc_id, _ ,tf in group] )

    result[k_group] = [(doc_id, tf) for doc_id, _ ,tf in group] 

pickle.dump(result , sys.stdout.buffer)  
	

'''lambda x: itemgetter(0)( itemgetter(1)(x) )'''
