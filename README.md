#This is the repository for Quora Question Duplicate - Search Engine Architecture course project.

Problem source:
https://www.kaggle.com/c/quora-question-pairs


Proposal: 
1. stemmer 
clean the text, men=man, had = have, 

2. word2vec get word feature vector
https://code.google.com/archive/p/word2vec/

3. cluster and regression

 3.1 cluster for grouping words, and for later similarity question detection 
choose top related word with the words user input, search questions containing related words.

 3.2 regression for tell duplicate question:
add all word’s feature vector (e.g. 300 dimension), get 2 doc’s cosine distance. Rregression function: cosine dist => [0,1] 
Regression output will tell if two questions are duplicate


#2017.05.02

#Model trainning and testing, to get search engine's corpse:

python3 -m general_stemmer.stop_stem

python3 -m general_word2vec_train.word2vec_train

#size1<404288 size2<404288 size1>size2
python3 -m stemmer.stemmer_select --size1=100000 --size2=20000

python3 -m word2vec.sentence2vec_idf

python3 -m word2vec.sentence2vec_noidf

python3 -m word2vec.generate_final_train

python3 -m classifier.kneighbors

python3 -m classifier.predict

#LSTM mddel

cd lstm/lstm_py

#size<404288
python3 lstm.py --size=10000 

#web search engine usage:
#go to linserv1

1. create input files
./assignment4/reformat_all.sh

2. start mapreduce framework
python -m assignment3.workers

3. submit indexing tasks to mapreduce framework
python -m assignment4.start
4. start search engine backend

python -m assignment2.start 
5. start search engine frontend
python -m webapp.start


Now, you can visit http://linserv1.cims.nyu.edu:22750/ to start search.