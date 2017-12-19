import pandas as pd
from nltk.stem import *
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string, os
import argparse

stemmer1 = SnowballStemmer("english", ignore_stopwords=True)
split_return = lambda line_text, index: (line_text.split(","))[index]

def stem_cleaned_text (text):
	global stemmer1
	stop_words = set(stopwords.words("english"))  # load stopwords
	wordsToken = word_tokenize(text)
	wordsToken = filter(lambda x: x not in string.punctuation, wordsToken)
	cleaned_text = filter(lambda x: x not in stop_words, wordsToken)
	stemmed_words = map (lambda x: stemmer1.stem(x) , cleaned_text )
	return '\t'.join(stemmed_words)

def stem_data(data_dir, train_filename, test_filename, train_out_filename, test_out_filename, num1, num2):

	train_data_path = os.path.join(data_dir, train_filename)
	train_out_path = os.path.join(data_dir, train_out_filename)
	test_data_path = os.path.join(data_dir, test_filename)
	test_out_path = os.path.join(data_dir, test_out_filename)
	df_train = pd.read_table(train_data_path, header=0, sep=',')
	df_train_out = df_train.iloc[0:num1, :]
	#print(df_train_out.head())
	df_test = pd.read_table(test_data_path, header=0, sep=',')
	df_test_out = df_test.iloc[0:num2, :]
	#print(df.head())
	'''
	id = df['id']
	qid1 = df['qid1']
	qid2 = df['qid2']
	'''

	global stemmer1
	stop_words = set(stopwords.words("english"))  # load stopwords
	#for i in range(len(df.question1)):
	for i, row in tqdm(df_train_out.iterrows()):
		#df.iloc[i,3] = df.iloc[i,3].astype(object)
		#df.iloc[i, 3] = stem_cleaned_text(df.iloc[i, 3])
		#df.loc[[i], ['question1']] = df.loc[[i], ['question1']].applymap(lambda x: stem_cleaned_text(x))
		try:
			#df_train_out.set_value(i,"id", row["id"])
			#df_train_out.set_value(i, "qid1", row["qid1"])
			#df_train_out.set_value(i, "qid2", row["qid2"])
			df_train_out.set_value(i,'question1', stem_cleaned_text(row["question1"]))
			df_train_out.set_value(i, 'question2', stem_cleaned_text(row["question2"]))
			#df_train_out.set_value(i, "is_duplicate", row["is_duplicate"])
		except:
			df_train_out.drop(df_train_out.index[i])
			pass

	df_train_out.to_csv(train_out_path, index=False, sep=',', encoding='utf-8')

	for i, row in tqdm(df_test_out.iterrows()):
		#df.iloc[i,3] = df.iloc[i,3].astype(object)
		#df.iloc[i, 3] = stem_cleaned_text(df.iloc[i, 3])
		#df.loc[[i], ['question1']] = df.loc[[i], ['question1']].applymap(lambda x: stem_cleaned_text(x))
		try:
			#df_test_out.set_value(i,'test_id', row["test_id"])
			df_test_out.set_value(i,'question1', stem_cleaned_text(row["question1"]))
			df_test_out.set_value(i, 'question2', stem_cleaned_text(row["question2"]))
		except:
			df_test_out.drop(df_test_out.index[i])
			pass

	df_test_out.to_csv(test_out_path, index=False, sep=',', encoding='utf-8')


def get_par_dir():
	print(__file__)
	print(os.path.join(os.path.dirname(__file__), os.path.pardir))
	parpath = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
	print("parent path is: " ,parpath)
	datapath = os.path.join(parpath, 'data')
	return (parpath, datapath)


if __name__ == "__main__":
	# text = "This is a sample sentence, showing off the stop words filtration filtr samples that."
	parser = argparse.ArgumentParser()
	parser.add_argument("--size1")
	parser.add_argument("--size2")
	args = parser.parse_args()
	size1 = int(args.size1)
	size2 = int(args.size2)
	pardir, datadir = get_par_dir()
	stem_data(datadir, "train.csv", "test.csv", 'stem_out_train.csv', 'stem_out_test.csv', size1, size2) #read from datadir/test.csv, write into datadir/stem_out
	#stem_data(datadir , "test.csv", 'test_stem_out')

