import os, pickle, gensim
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
from sklearn import svm # no use!
import warnings
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier #0.89
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier #0.86
import matplotlib.pyplot as plt

def get_par_dir():
	print(__file__)
	parpath = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
	print("parent path is: " ,parpath)
	datapath = os.path.join(parpath, 'data')
	return (parpath, datapath)

def training_square_error(y, predicts):
	sum = 0
	correct = 0
	correct_rate = 0.0
	y = y.tolist()
	for i in range(len(y)):
		if (predicts[i]*y[i]) < 0:
			#print(type(predicts[i]))
			sum += (predicts[i] - y[i])**2
		else:
			correct += 1
	MSE = sum / float(len(y))
	#print(type(MSE))
	correct_rate = correct / float(len(y))
	print("MSE: ", MSE)
	print("Training Correct Rate: ", correct_rate)

# classifier based on LinearRegression:
def train(data_dir, train_name, test_name):
	warnings.filterwarnings("ignore")
	train_path = os.path.join(data_dir, train_name)
	test_path = os.path.join(data_dir, test_name)
	df_train = pd.read_csv(train_path,header=0)
	df_test = pd.read_csv(test_path,header=0)
	#print(type(df.iloc[0,:]))
	#clf_name_list = ["RandomForest", "AdaBoost", "DecisionTree"]
	#clf1 = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
	#clf2 = AdaBoostClassifier()
	clf = DecisionTreeClassifier(max_depth=5)
	#clf_list = [clf1, clf2, clf3]
	#print(type(df.iloc[:, 0].values))
	#print(df.iloc[:, 1].values)
	X_train = df_train[["len_diff_perc","com_perc","similarity_idf","similarity_noidf"]].values
	#X_test = df_test[["len_diff_perc","com_perc","similarity_idf","similarity_noidf"]].values
	#print(X[0:5])
	#print(X.shape)
	y_train = df_train[["is_duplicate"]].values
	#y = y.ravel()
	y_train = y_train.ravel()
	y_train = np.array(y_train).astype(int)
	#X = np.array([[0.2], [0.3], [0.4], [1], [2], [3], [4]])
	#y = np.array([0, 0, 0, 1, 1, 1, 1])
	#plt.plot(X,y, 'ro')
	#for i in range(len(X)):
	#	plt.annotate(i, (X[i], y[i]))
	#plt.show()
	#X = X.reshape(-1,1)
	#x_axis = [i for i in range(len(X))]
	#plt.plot(x_axis, y, 'r*')
	#y = y.reshape(-1,1)
	#print(y.shape)
	#print(X.shape)
	#est.fit(X,y)
	#res_list = []
	#for name, clf in zip(clf_name_list, clf_list):
	print("ready to fit: ")
	clf.fit(X_train, y_train)
		#print(clf.support_vectors_)
		#print(clf.coef_)
		#print(clf.intercept_)
	print("fit success!")
	predicts = []
	for i in range(10000):
		print("I'm predict: ", i)
		predicts.append(clf.predict(X_train[i])[0])
		#print(est.predict(np.array([0.3])))
		# ref: http://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html
		#print('Variance score: %.2f' % clf.score(X, y))
		#plt.plot(x_axis,predicts,'bo')
		#print(est.intercept_)
		#plt.show()
	training_square_error(y_train[0:10000], predicts)
	#res_list.append(clf)
	return clf

if __name__ == '__main__':
	pardir, datadir = get_par_dir()
	clf = train(datadir, "class_train_final.csv", "class_test_final.csv")
	pickle.dump(clf, open(os.path.join(datadir, "decisiontree.pkl"), 'wb'))

