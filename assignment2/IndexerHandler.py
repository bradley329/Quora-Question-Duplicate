# return a posting list for query word:[(id, score)]
import tornado.httpserver
import tornado.ioloop
import tornado.web
import tornado.httpclient
import tornado.gen
import tornado.options
from tornado.escape import json_decode,json_encode
from nltk.tokenize import RegexpTokenizer
from collections import deque
import socket , json, nltk, pickle, os, sys
import xml.etree.ElementTree as ET
import numpy as np
from assignment2.inventory import *
from itertools import groupby

# inner_prod = lambda a,b: sum([aa*bb for aa,bb in (a,b)])
def inner_prod(a,b):
        ret_sum = 0
        for i in range(len(a)):
                ret_sum+= a[i]*b[i]
        return ret_sum
def createOneVector(dim, onehot, val):
        ret = [0]*dim
        ret[onehot] = val
        return ret

def sumvector(listvec):
        return sum([np.array(x) for x in listvec])
'''def query_tf_vector(qarr, ret_q): #qarr is query original arr, ret_q is valid q for searching
        dim = len(ret_q)
        ret = [0]* dim
        for i in range(dim):
                ret[i] += qarr.count(ret_q[i])
        return ret'''

class IndexerHandler(tornado.web.RequestHandler):
        idf_path = JOB_PATHS['idf']  #IDF is from idf_jobs/0.out
        for f in read_output(idf_path):
                with open(f, 'rb') as idf_fd:
                        IDF = pickle.load(idf_fd)
        INDEXER_ID = 0

        def initialize(self, port):

                self.port = port
                partition_idx = INDEX_PORTS.index(self.port)
                with open( os.path.join( JOB_PATHS['invindex'], "%d.out"%partition_idx) , 'rb' ) as inv_fd:
                        self.inv_dict = pickle.load(inv_fd)
        @staticmethod
        def getQueryIDF(qarr):
                ret_q = []
                ret_tf = []
                ret_idf = []
                group_qarr = groupby(sorted(qarr))
                for q, tf_q in group_qarr:
                        if IndexerHandler.IDF.get(q):
                                ret_q.append(q)
                                ret_tf.append(  len(list(tf_q)) )
                                ret_idf.append(IndexerHandler.IDF[q] )
                return (ret_q,ret_tf,ret_idf)

        @tornado.gen.coroutine
        def get(self):
                resp = []
                # q = self.get_argument("q")
                qs = self.get_argument("q")  # /index?q=query_here
                qs = qs.lower()
                qarr = qs.split(" ")

                qarr = [q.strip() for q in qarr]

                (ret_q,q_tfs,idfs) = IndexerHandler.getQueryIDF(qarr) #determine dimension of ret_q, maximum

                if len(ret_q) == 0:
                        resp = json.dumps({"postings": []})
                else:

                        # idf = IndexerHandler.IDF[q]   # when word's q_df close to page_num, its socre is 0
                        # check each partition of invindex
                        dim = len(ret_q) #[1,1,1,1] every word is one in vector
                        tfs = {} # every doc_id has a list of vector
                        for i in range(dim): # for every dim of query string list
                                valid_q = ret_q[i]
                                if self.inv_dict.get(valid_q): #can find q in inv_dict
                                        for (docID, tf) in self.inv_dict[valid_q]:
                                                if( not tfs.get(docID)):
                                                        tfs[docID] = []
                                                tfs[docID].append( createOneVector(dim, i, int(tf) ) )
                        doc_tfidf_vector = {}
                        score = {}
                        final_tf = {}
                        for docID in tfs.keys():
                                final_tf[docID] = sumvector(tfs[docID]) # a list of vector

                        idfs_arr = np.array(idfs)
                        query_tfidf_vector = idfs_arr * np.array(q_tfs)
                        print("query vector", ret_q)
                        print("query tf vector", q_tfs)
                        print("query idf words", idfs)

                        for docID in final_tf:
                                doc_tfidf_vector[docID] =  idfs_arr* np.array(final_tf[docID])
                                score[docID] = np.dot(doc_tfidf_vector[docID], query_tfidf_vector)
                                # print("docID: ",docID,"doc_tfidf: ",doc_tfidf_vector[docID], "score: ", score[docID] )
                                resp.append([docID, score[docID]])

                        sorted_resp = sorted(resp, key=lambda tp: tp[1],  reverse = True)
                        resp = json.dumps({"postings": sorted_resp},sort_keys=True)

                self.write(resp) #encode result to json



if __name__ == "__main__":
        #depend nodelist in PageNode
        apps = {}
        for index_port in INDEX_PORTS:
                url = BASE_ADDR  % index_port
                print("Indexer%d url: %s" %(IndexerHandler.INDEXER_ID, url))
                IndexerHandler.INDEXER_ID+=1
                apps[index_port] = tornado.web.Application(handlers=[(r"/index", IndexerHandler, dict(port = index_port))],debug = True)
                apps[index_port].listen(index_port)

        tornado.ioloop.IOLoop.instance().start()

#IDF is word's  DocumentFrequency [word, IDF] pair
#
