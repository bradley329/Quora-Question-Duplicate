# since we've known that the results from the KNN model works the best,
# we choose the KNN model and perform the prediction
import os, pickle, gensim
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
import warnings
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier #0.89
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

def get_par_dir():
	print(__file__)
	parpath = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
	print("parent path is: " ,parpath)
	datapath = os.path.join(parpath, 'data')
	return (parpath, datapath)

def predict(clf, data_dir,test_name,original_name, out1_name, out2_name):
	warnings.filterwarnings("ignore")
	original_path = os.path.join(data_dir, original_name)
	test_path = os.path.join(data_dir, test_name)
	out1_path = os.path.join(data_dir, out1_name)
	out2_path = os.path.join(data_dir, out2_name)
	df_original = pd.read_csv(original_path, header=0)
	df_test = pd.read_csv(test_path, header=0)
	X_test = df_test[["len_diff_perc", "com_perc", "similarity_idf", "similarity_noidf"]].values
	predicts = clf.predict(X_test)
	df_test["is_duplicate"] = pd.Series(predicts)
	pos = 0
	for i in range(len(predicts)):
		if predicts[i]==1:
			pos += 1
	print("num of duplicate: ", pos)
	df_test.to_csv(out1_path , index=False,encoding='utf-8')
	# merge original and df_test:
	df_original["is_duplicate"] = pd.Series(predicts)
	df_original.to_csv(out2_path, index=False,encoding='utf-8')

if __name__ == '__main__':
	pardir, datadir = get_par_dir()
	clf = pickle.load(open(os.path.join(datadir, "kneighbors.pkl"), 'rb'))
	predict(clf, datadir, "class_test_final.csv", "test_20000.csv", "out/test_predict_stem.csv","out/test_predict_nonstem.csv")


