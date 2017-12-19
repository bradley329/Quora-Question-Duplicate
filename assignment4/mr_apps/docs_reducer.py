#!/usr/bin/env python

from itertools import groupby
from operator import itemgetter
import sys,pickle

result = {}
# p.stdin.write(('%s\t%s\n' % (pair[0], pair[1])).encode())

data = map(lambda x: x.strip().split('\t',3), sys.stdin)

for doc_id, question1, question2, isDup in data:
	question1 = eval('%s' % question1)
	question2 =  eval('%s' % question2)
	result[doc_id] = (question1, question2, isDup) #dict from doc_id->(question1, question2, isDup)

pickle.dump(result, sys.stdout.buffer)