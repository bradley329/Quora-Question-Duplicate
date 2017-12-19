import os, pickle, gensim
import numpy as np
import pandas as pd
from tqdm import tqdm
import json

# combine the taining set which did not take tfidf into consideration and which takes tfidf into consideration
def get_par_dir():
	print(__file__)
	parpath = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
	print("parent path is: " ,parpath)
	datapath = os.path.join(parpath, 'data')
	return (parpath, datapath)

def generate_train(data_dir, original, data1_name, data2_name, out_name, original_out):
	original_path = os.path.join(data_dir, original)
	original_out_path = os.path.join(data_dir, original_out)
	df_original = pd.read_csv(original_path, header=0)
	df_original = df_original.iloc[0:100000, :]
	df_original.to_csv(original_out_path, index=False, encoding='utf-8')
	data1_path = os.path.join(data_dir, data1_name)
	df_idf = pd.read_csv(data1_path, header=0)
	data2_path = os.path.join(data_dir, data2_name)
	df_noidf = pd.read_csv(data2_path, header=0)
	out_path = os.path.join(data_dir, out_name)
	#df = pd.DataFrame(columns=["len_diff_perc", "com_word_perc", "similarity_noidf","similarity_idf","is_duplicate"])
	# combine the two dataframe together
	df = pd.merge(df_noidf, df_idf, on=["id","is_duplicate"])
	'''
	for i in range(len(df_noidf["similarity"])):
		
		similarity_noidf = df_noidf.iloc[i,0]
		is_duplicate = df_noidf.iloc[i,1]
		len_diff_perc = df_idf.iloc[i,0]
		com_word_perc = df_idf.iloc[i,1]
		similarity_idf = df_idf.iloc[i,2]
		df.set_value(i, "len_diff_perc", len_diff_perc)
		df.set_value(i, "com_word_perc", com_word_perc)
		df.set_value(i, "similarity_noidf", similarity_noidf)
		df.set_value(i, "similarity_idf", similarity_idf)
		df.set_value(i, "is_duplicate", int(is_duplicate))
		'''
	df.to_csv(out_path, index=False,encoding='utf-8')

def generate_test(data_dir, original, data1_name, data2_name, out_name, original_out):
	original_path = os.path.join(data_dir, original)
	original_out_path = os.path.join(data_dir, original_out)
	df_original = pd.read_csv(original_path, header=0)
	df_original = df_original.iloc[0:20000, :]
	df_original.columns = ["id", "question1","question2"]
	df_original.to_csv(original_out_path, index=False, encoding='utf-8')
	data1_path = os.path.join(data_dir, data1_name)
	df_idf = pd.read_csv(data1_path, header=0)
	data2_path = os.path.join(data_dir, data2_name)
	df_noidf = pd.read_csv(data2_path, header=0)
	out_path = os.path.join(data_dir, out_name)
	#df = pd.DataFrame(columns=["len_diff_perc", "com_word_perc", "similarity_noidf","similarity_idf","is_duplicate"])
	# combine the two dataframe together
	df = pd.merge(df_noidf, df_idf, on=["id"])
	df.to_csv(out_path, index=False,encoding='utf-8')
if __name__ == '__main__':
	pardir, datadir = get_par_dir()
	generate_train(datadir, "train.csv", "train_idf.csv", "train_noidf.csv", "class_train_final.csv", "train_100000.csv")
	generate_test(datadir, "test.csv", "test_idf.csv", "test_noidf.csv", "class_test_final.csv", "test_20000.csv")