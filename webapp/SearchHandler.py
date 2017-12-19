import tornado.httpserver
import tornado.ioloop
import tornado.web
import tornado.httpclient
import tornado.gen
import tornado.options
import urllib
from assignment2.inventory import *
import json
from tornado.escape import json_decode, json_encode

class IndexHandler(tornado.web.RequestHandler):
    def initialize(self, _port):
        print("initializing on port ", _port)
        self.port = _port 
    
    def get(self):
        self.render("index.html") 

class SearchHandler(tornado.web.RequestHandler):
    @tornado.gen.coroutine
    def post(self):
        q1_keepcase = (self.get_argument("question1", None))
        q2_keepcase = (self.get_argument("question2", None))
        question1 = q1_keepcase.strip().lower()
        question2 = q2_keepcase.strip().lower()
        q_cat = " ".join([question1,question2])

        # visit http://localhost:22749/search?q=q_cat
        params = urllib.parse.urlencode({'q': q_cat})
        # print("params:", params)
        url = "%s/search?%s" % (BASE_ADDR%org_port, params)
        print("searchURL:", url)
        http_async = tornado.httpclient.AsyncHTTPClient()
        futures = http_async.fetch(url)
        # print(dir(futures))
        r = yield futures
        resp = json.loads(r.body.decode() )
        num_results = resp["num_results"]
        searchResults = resp["results"]
        # self.write(resp)
        self.render("result.html" , searchResults = searchResults, title = "search results" ,
            question1 = q1_keepcase, question2 = q2_keepcase)



