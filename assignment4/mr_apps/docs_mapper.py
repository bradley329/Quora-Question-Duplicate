#!/usr/bin/env python

from itertools import groupby
from operator import itemgetter
import sys,pickle

data = pickle.load(sys.stdin.buffer)

doc_store = {}
for doc_id in data.keys():
	# doc_store[doc_id] = data[doc_id]
	print("%s\t%r\t%r\t%s" % ( doc_id, data[doc_id][0],data[doc_id][1], data[doc_id][2] ) )
	#eval('%s'%sp[1])