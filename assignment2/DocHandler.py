# Doc Handler.py

import tornado.httpserver
import tornado.ioloop
import tornado.web
import tornado.httpclient
import tornado.gen
import tornado.options
from tornado.escape import json_decode,json_encode
import json, hashlib, pickle, sys, os

from assignment2.IndexerHandler import IndexerHandler
from assignment2.inventory import *




class DocHandler(tornado.web.RequestHandler):
	DOCSERVER_ID = 0

	def initialize (self, port):
		self.port = port
		self.pageList = [] # store pages partition

		idx = INDEX_PORTS.index(self.port)
		with open( os.path.join( JOB_PATHS['docs'] , "%d.out" % idx), 'rb' ) as doc_fd:
				self.doc_dict = pickle.load(doc_fd)
	@staticmethod
	def createSnippet(q, text):
		index_q = text.casefold().find(q.casefold())
		len_q = len(q)
		start = max(0, index_q - 100)
		end = min(len(text)-1, index_q+100)
		ret = text[start:end]
		q_in_text = text[index_q:index_q+len_q]
		ret = ret.replace( q_in_text, "<strong>" + q_in_text + "</strong>")
		return ret

	def get(self):
		id = self.get_argument("id")
		q = self.get_argument("q")
		score = self.get_argument("score")

		partition_idx = PARTITIONER(id, NUM_DOC_PART)
		if INDEX_PORTS[partition_idx] == self.port:
			result = {}
			result["id"] = id
			result["question1"] = self.doc_dict[id][0] #question1
			# result["url"] = page_found.url
			result["question2"] = self.doc_dict[id][1] #question2
			result["isDup"] = self.doc_dict[id][2]
			result["score"] = score
			# result["tf_idf"] = self
			resp = json.dumps({"results": result},sort_keys=True)
			self.write(resp)

if __name__ == "__main__":
	apps = {}
	for doc_port in DOC_PORTS:
		url = BASE_ADDR % doc_port
		# print("Doc Handler%d url: %s" %(DocHandler.DOCSERVER_ID, url))
		DocHandler.DOCSERVER_ID+=1
		apps[doc_port] = tornado.web.Application(handlers=[(r"/doc", DocHandler, dict(port = doc_port))],debug = True)
		apps[doc_port].listen(doc_port)

	tornado.ioloop.IOLoop.instance().start()

