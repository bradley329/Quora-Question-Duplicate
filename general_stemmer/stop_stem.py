# -*- coding: utf-8 -*-
# clean stop word and stemming text, then pickled cleaned text to files
# data structure is

'''
PICKLE DATASTRUCTURE:
key : id 
value:
	dict:
	question1
	question2
	is_duplicate : 0 or 1
'''
# id - the id of a training set question pair
# qid1, qid2 - unique ids of each question (only available in train.csv)
# question1, question2 - the full text of each question
# is_duplicate - the target variable, set to 1 if question1 and question2 have essentially the same meaning, and 0 otherwise.

#  e.g. "11","23","24","How do I read and find my YouTube comments?","How can I see all my Youtube comments?","1"

from nltk.stem import *
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string, pickle , os, sys, json, csv
import pandas as pd


stemmer1 = SnowballStemmer("english", ignore_stopwords=True)
split_return = lambda line_text, index: (line_text.split(","))[index]

def stem_cleaned_text (text):
	global stemmer1
	text = text.strip('"')
	stop_words = set(stopwords.words("english"))  # load stopwords
	wordsToken = word_tokenize(text) 
	wordsToken = filter(lambda x: x not in string.punctuation, wordsToken)
	cleaned_text = filter(lambda x: x not in stop_words, wordsToken)
	stemmed_words = map (lambda x: stemmer1.stem(x) , cleaned_text )
	return list(stemmed_words)

def stem_file(data_dir, data_filename, out_filename):
	#read infile, and pickle to outfile
	res = []
	data_path = os.path.join( data_dir, data_filename)
	out_path  =  os.path.join( data_dir, out_filename) 
	out_file = os.path.join( data_dir, out_filename+'_f')
	df = pd.read_csv(data_path)
	ids = list(df['id'])
	questions1 = list(df['question1'])
	questions2 = list(df['question2'])
	is_dup = list(df['is_duplicate'])
	with open(out_path, 'wb')  as outfile:
		for i in range(len(questions1)):
			curID = ids[i]
			q1 = str(questions1[i])
			#print(q1)
			q2 = str(questions2[i])
			if len(q1) == 0 or len(q2) == 0:
				continue
			is_duplicate_bit = int(is_dup[i])
			stemmed_q1 = stem_cleaned_text(q1)
			stemmed_q2 = stem_cleaned_text(q2)
			res.append((curID, stemmed_q1, stemmed_q2, is_duplicate_bit))
		labels = ['id', 'q1', 'q2', 'is_duplicate']
		df_res = pd.DataFrame.from_records(res, columns = labels)
		df_res.to_csv('data/stemmed_out.csv', sep = '\t')
		#pickle.dump(ret_dict, outfile)
		#outfile_f.write( json.dumps(ret_dict, indent =2 ) )
		#outfile_f.close()


def get_par_dir():
	print(__file__)
	parpath = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
	print("parent path is: " ,parpath)
	datapath = os.path.join(parpath, 'data')
	return (parpath, datapath)


if __name__ == "__main__":
	# text = "This is a sample sentence, showing off the stop words filtration filtr samples that."
	pardir, datadir = get_par_dir()
	stem_file(datadir , "train.csv", 'stem_out') #read from datadir/test.csv, write into datadir/stem_out

