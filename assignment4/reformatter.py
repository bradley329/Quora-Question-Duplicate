import os
from os import listdir
from os.path import isfile, join
import sys, urllib
import socket
import json
import xml.etree.ElementTree as ET
import argparse, pickle, csv

def clean_old_in(job_path):
	file_list  = listdir(job_path)
	out_list = [f for f in file_list if (isfile(join(job_path, f)) and f[-3:] == ".in") ]
	for of in out_list:
		os.remove(join(job_path, of))

if __name__ == "__main__":
	train_path = "data/out/test_predict_nonstem.csv"
	pickle_dir = "data/out"
	# id,question1,question2,is_duplicate
	with open(train_path, mode='r') as infile:
		reader = csv.reader(infile)
		linenum = 20000
		mydict = {row[0]:(row[1], row[2], row[3]) for row in reader if row[0]!='id' and int(row[0])<linenum}
		# print( len( mydict.keys() ))
		print(mydict["2"])

	pickle_path = os.path.join(pickle_dir, "train_pickle")
	all_fileds_fd = open(pickle_path, 'wb')
	# pickle.dump( mydict, all_fileds_fd )
	# mydict[id]->tuple('How can I increase the speed of my internet connection while using a VPN?', 'How can Internet speed be increased by hacking through DNS?', '0')

	parser = argparse.ArgumentParser(description='Process quora question.')
	parser.add_argument('--job_path')
	parser.add_argument('--num_partitions' )
	args = parser.parse_args()

	num_partitions = int(args.num_partitions)
	job_path =args.job_path

	lines = [0]*num_partitions
	clean_old_in(job_path)
	fds = [open( join( job_path+'%d.in'%index), 'wb') for index in range(num_partitions)] #binary form output
	page_entries = [{} for i in range(num_partitions)]
	
	for id in mydict.keys():
		idx = int(id) %  num_partitions
		# mydict[id][0] is question1, mydict[id][1] is question2
		page_entries[idx][id] = (mydict[id][0], mydict[id][1] , mydict[id][2])	
		# page_entries[idx][page.id] =  page.text	
		lines[idx] +=1

	for idx in range(num_partitions):
		pickle.dump(  page_entries[idx] ,  fds[idx] )
		fds[idx].close()

	print("num_lines: ",lines)
