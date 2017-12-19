# inventory.py
# SERVER_PATTERN = "http://linserv2.cims.nyu.edu:%d"
# SERVER_PATTERN = "http://localhost:%d"
import assignment2.inventory
SERVER_PATTERN = assignment2.inventory.BASE_ADDR


START_PORT = 22905
WORKER_NUM = 20
MAPPER_MAXNUM = 8
REDUCER_MAXNUM = 8
PORT_LIST = list( map(lambda i: START_PORT+10*i, range(WORKER_NUM)) )
