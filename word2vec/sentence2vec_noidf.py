import os, pickle, gensim
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import preprocessing
import json

# this code is to do a feature map for sentences
# after we've got the vector representations of words
def get_par_dir():
	print(__file__)
	parpath = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
	print("parent path is: " ,parpath)
	datapath = os.path.join(parpath, 'data')
	return (parpath, datapath)

# Inner product based on the similarity of words:
# vec1 = [1,2,3,4,5,...300]; vec2 = [300,299,...1]
#def inner(vec1, vec2):
# note this only trained the 100000 training data
def train(data_dir, data_name, model_name, out_name):
	model_path = os.path.join(data_dir, model_name)
	data_path = os.path.join(data_dir, data_name)
	df = pd.read_csv(data_path, header=0)
	df['question1'] = df['question1'].apply(lambda x: str(x))
	df['question1'] = df['question1'].apply(lambda x: x.split("\t"))
	df['question2'] = df['question2'].apply(lambda x: str(x))
	df['question2'] = df['question2'].apply(lambda x: x.split("\t"))
	q1 = list(df['question1'])
	q2 = list(df['question2'])
	is_dup = list(df['is_duplicate'])
	out_path = os.path.join(data_dir, out_name)
	model = gensim.models.Word2Vec.load(model_path)
	res = []
	#print(model["me"])
	#print(model["you"])
	#print(np.dot(model["me"], model['you']))
	#print(model.similarity('me', 'you'))

	
	#questions1 = []
	#questions2 = []

	for qID in tqdm(range(len(q1))):
		words1 = q1[qID]
		words2 = q2[qID]
		if len(words1) == 0 or len(words2) == 0:
			continue
		is_duplicate = int(is_dup[qID])
		i = 0
		sentence1_vec = np.zeros(len(model["me"]))
		sentence2_vec = np.zeros(len(model["me"]))
		for word in words1:
			#print("does this word valid in the model? QID is: \n", word, qID)
			try:
				word_vector = model[word]
				for j in range(len(word_vector)):
					sentence1_vec[j] += word_vector[j]
				i += 1
			except:
				pass

		# calculate the mean for each dimension:
		if i == 0:
			continue
		for j in range(len(model["me"])):
			sentence1_vec[j] = sentence1_vec[j] / i

		sentence1_vec = preprocessing.normalize(sentence1_vec.reshape(1, -1), norm='l2')
		sentence1_vec = sentence1_vec[0]
		i = 0
		for word in words2:
			#print("does this word valid in the model? QID is: \n", word, qID)
			try:
				word_vector = model[word]
				for j in range(len(word_vector)):
					sentence2_vec[j] += word_vector[j]
				i += 1
			except:
				pass
		# calculate the mean:
		if i == 0:
			continue
		for j in range(len(model["me"])):
			sentence2_vec[j] = sentence2_vec[j] / i

		sentence2_vec = preprocessing.normalize(sentence2_vec.reshape(1, -1), norm='l2')
		sentence2_vec = sentence2_vec[0]
		#print(sentence1_vec,' ',sentence2_vec,'\n')
		similarity = np.dot(sentence1_vec, sentence2_vec)
		if int(is_duplicate)==0:
			is_duplicate = -1
		res.append((qID, similarity, is_duplicate))
		#k += 1
		#if k == 100:
		#	break
	labels = ['id', 'similarity_noidf', 'is_duplicate']
	res = pd.DataFrame.from_records(res, columns = labels)

	print("processing job done")
	res.to_csv(out_path, index=False)
	# pickle the res dict to a new pickle file:
	#pickle.dump(res, open(out_path, 'wb'))
	# when performing dump got killed error due to memory limitations

def test(data_dir, data_name, model_name, out_name):
	model_path = os.path.join(data_dir, model_name)
	data_path = os.path.join(data_dir, data_name)
	df = pd.read_csv(data_path, header=0)
	df['question1'] = df['question1'].apply(lambda x: str(x))
	df['question1'] = df['question1'].apply(lambda x: x.split("\t"))
	df['question2'] = df['question2'].apply(lambda x: str(x))
	df['question2'] = df['question2'].apply(lambda x: x.split("\t"))
	q1 = list(df['question1'])
	q2 = list(df['question2'])
	#is_dup = list(df['is_duplicate'])
	out_path = os.path.join(data_dir, out_name)
	model = gensim.models.Word2Vec.load(model_path)
	res = []
	# print(model["me"])
	# print(model["you"])
	# print(np.dot(model["me"], model['you']))
	# print(model.similarity('me', 'you'))


	# questions1 = []
	# questions2 = []

	for qID in tqdm(range(len(q1))):
		words1 = q1[qID]
		words2 = q2[qID]
		if len(words1) == 0 or len(words2) == 0:
			continue
		#is_duplicate = int(is_dup[qID])
		i = 0
		sentence1_vec = np.zeros(len(model["me"]))
		sentence2_vec = np.zeros(len(model["me"]))
		for word in words1:
			# print("does this word valid in the model? QID is: \n", word, qID)
			try:
				word_vector = model[word]
				for j in range(len(word_vector)):
					sentence1_vec[j] += word_vector[j]
				i += 1
			except:
				pass

		# calculate the mean for each dimension:
		if i == 0:
			continue
		for j in range(len(model["me"])):
			sentence1_vec[j] = sentence1_vec[j] / i

		sentence1_vec = preprocessing.normalize(sentence1_vec.reshape(1, -1), norm='l2')
		sentence1_vec = sentence1_vec[0]
		i = 0
		for word in words2:
			# print("does this word valid in the model? QID is: \n", word, qID)
			try:
				word_vector = model[word]
				for j in range(len(word_vector)):
					sentence2_vec[j] += word_vector[j]
				i += 1
			except:
				pass
		# calculate the mean:
		if i == 0:
			continue
		for j in range(len(model["me"])):
			sentence2_vec[j] = sentence2_vec[j] / i

		sentence2_vec = preprocessing.normalize(sentence2_vec.reshape(1, -1), norm='l2')
		sentence2_vec = sentence2_vec[0]
		# print(sentence1_vec,' ',sentence2_vec,'\n')
		similarity = np.dot(sentence1_vec, sentence2_vec)
		#if int(is_duplicate) == 0:
		#	is_duplicate = -1
		res.append((qID, similarity))
	# k += 1
	# if k == 100:
	#	break
	labels = ['id', 'similarity_noidf']
	res = pd.DataFrame.from_records(res, columns=labels)

	print("processing job done")
	res.to_csv(out_path, index=False, encoding='utf-8')


# pickle the res dict to a new pickle file:
# pickle.dump(res, open(out_path, 'wb'))
# when performing dump got killed error due to memory limitations

if __name__ == '__main__':
	pardir, datadir = get_par_dir()
	train(datadir, "stem_out_train.csv", "word2vec.mdl", "train_noidf.csv")
	test(datadir,"stem_out_test.csv", "word2vec.mdl", "test_noidf.csv")
