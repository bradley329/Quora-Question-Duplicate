# usage: python -m webapp.start 8888
import tornado.ioloop
import tornado.web
import sys, socket
from webapp.SearchHandler import *
import assignment2.inventory
if __name__ == "__main__":
	# print("argv", sys.argv[1])
	listen_port =  22750
	hostIP = assignment2.inventory.BASE_ADDR
	IPaddr = "%s:%d"
	print("start search engine on", IPaddr%(hostIP,listen_port))
	app = tornado.web.Application(handlers= [(r"/", IndexHandler, dict(_port = listen_port) ), (r"/searchresult", SearchHandler)],debug = True)
	app.listen(listen_port)
	tornado.ioloop.IOLoop.current().start()