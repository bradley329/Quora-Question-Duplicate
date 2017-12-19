from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy
import os
from sklearn.cross_validation import train_test_split
from keras.preprocessing.sequence import pad_sequences
import data_process
import pickle

BATCH_SIZE = 32

def vectorize_pair(pair, word2index, seq_maxlen):
	x_ques1 = []
	x_ques2 = []
	for w in pair[0]:
		if w in word2index:
			x_ques1.append(word2index[w])
		else:
			x_ques1.append(len(word2index) / 2)
	for w in pair[1]:
		if w in word2index:
			x_ques2.append(word2index[w])
		else:
			x_ques2.append(len(word2index) / 2)
	x1 = [x_ques1]
	x2 = [x_ques2]
	# x_ques1.append([word2idx[w] for w in pair[0] if w in word2idx else len(word2index)])
	# x_ques2.append([word2idx[w] for w in pair[1] if w in word2idx else len(word2index)])
	x1 = pad_sequences(x1, maxlen=seq_maxlen)
	x2 = pad_sequences(x2, maxlen=seq_maxlen)
	return x1, x2

def parse_input(q1, q2):
	return q1.strip('"').split(" "), q2.strip('"').split(" ")

def predict(q1, q2):
	print("data process...")
	pair = parse_input(q1, q2)
	# pair = data_process.parse_quora_dul_data("a")
	word2idx = pickle.load(open("../data/word2index/word2index.pkl", 'rb'))
	seq_maxlen = pickle.load(open("../data/seq_maxlen/seq_maxlen.pkl", "rb"))
	# x_ques1, x_ques2, y, pids = data_process.vectorize_ques_pair(pair, word2idx, seq_maxlen)
	# x_ques1train, x_ques1test, x_ques2train, x_ques2test, ytrain, ytest, pidstrain, pidstest = train_test_split(x_ques1, x_ques2, y, pids, test_size=0.2, random_state=42)
	x_ques1, x_ques2 = vectorize_pair(pair, word2idx, seq_maxlen)

	# Load model from previous train
	json_format_model = open("../data/quora_dul_lstm.json", "r")
	loaded_model_json = json_format_model.read()
	json_format_model.close()
	model = model_from_json(loaded_model_json)
	model.load_weights("../data/quora_dul_best_lstm.hdf5")
	print("Load model from previous train")

	# Evaluate loaded model on test data
	model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

	# Predict
	print ("predict...")
	y_test_pred = model.predict_classes([x_ques1, x_ques2], batch_size=BATCH_SIZE)
	# print(y_test_pred)

	# loss, acc = model.evaluate([x_ques1test, x_ques2test], ytest, batch_size=BATCH_SIZE)
	# print("Test loss/accuracy final model = %.4f, %.4f" % (loss, acc))
if __name__ == '__main__':
	predict("Who is he?", "Does dog eat fish?")