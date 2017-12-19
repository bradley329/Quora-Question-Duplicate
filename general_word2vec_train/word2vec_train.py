# -*- coding: utf-8 -*-
import os, pickle, gensim, pickle
import pandas as pd
from tqdm import tqdm

def get_par_dir():
	print(__file__)
	parpath = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
	print("parent path is: " ,parpath)
	datapath = os.path.join(parpath, 'data')
	return (parpath, datapath)

def train(data_dir, data_filename, out_filename):
	data_path = os.path.join(data_dir, data_filename)
	out_path = os.path.join(data_dir, out_filename)
	# stemmed_data = pickle.load(open(data_path, 'rb'), encoding = 'utf-8')
	df = pd.read_csv(data_path, sep = '\t')
	df['q1'] = df['q1'].apply(lambda x: str(x))
	df['q1'] = df['q1'].apply(lambda x: eval(x))
	df['q2'] = df['q2'].apply(lambda x: str(x))
	df['q2'] = df['q2'].apply(lambda x: eval(x))
	ids = list(df['id'])
	q1 = list(df['q1'])
	q2 = list(df['q2'])

	questions = q1 + q2

	print(questions[0])
	print(questions[1])

	if os.path.exists('data/word2vec.mdl'):
		model = gensim.models.KeyedVectors.load_word2vec_format('data/word2vec.bin', binary = True)
		model.init_sims(replace = True)
		temp = dict(zip(model.index2word, model.syn0))
		print("Number of tokens in word2vec:", len(temp.keys()))
	else:
		print("training")
		model = gensim.models.Word2Vec(questions, size = 300, workers = 16, iter = 10, negative = 20)
		print("training done")
		model.init_sims(replace = True)
		try:
			temp = dict(zip(model.wv.index2word, model.wv.syn0))
			print("Number of tokens in word2vec:", len(temp.keys()))
		except:
			pass
		model.save('data/word2vec.mdl')
		model.wv.save_word2vec_format('data/word2vec.bin', binary = True)
		del questions

if __name__ == '__main__':
	pardir, datadir = get_par_dir()
	train(datadir, "stemmed_out.csv", "word2vec.mdl")
