import os, json, urllib, hashlib
# org_port = 21173
import socket
org_port = 22749

STEP_INDEX = 10


hostIP = socket.gethostbyname(socket.gethostname())

# BASE_ADDR = hostIP+":%d"
BASE_ADDR = 'http://linserv1.cims.nyu.edu:%d'
# BASE_ADDR = 'http://localhost:%d'
#if cannot visit website by host's IP, hardcode BASE_ADDR(include http://)

print("assignment2.inventory: initialize on host IP - %s" % hostIP)


INDEXER_NUM  = 7
INDEX_PORTS =list( map(lambda i: org_port+STEP_INDEX*(i+1), range(INDEXER_NUM)))
INDEX_PATT = BASE_ADDR+"/index?q=%s"
# params = urllib.parse.urlencode({'reducer_ix': reducer_ix, 'map_task_id': map_task_id_list[i]})
# url = "http://%s/retrieve_map_output?%s" % (server, params)
# INDEX_PATT = "http://localhost:%d/index?q=%s"

STEP_DOC = 9
NUM_DOC_PART = 7
DOC_PORTS =list( map(lambda i: org_port+STEP_DOC*(i+1), range(NUM_DOC_PART)))

DOC_PATT = BASE_ADDR+"/doc?id=%s&q=%s"
JOB_NAMES = ['invindex', 'docs', 'idf' ]
MR_OUTPUT_PATH = 'assignment4'
JOB_PATHS={}

for job in JOB_NAMES:
	JOB_PATHS[job] =os.path.join(MR_OUTPUT_PATH, '%s_jobs' % job)

PARTITIONER = lambda key, num_reducers: int(hashlib.md5(key.encode()).hexdigest()[:8], 16) % num_reducers
read_output = lambda job_path: [os.path.join(job_path,f) for f in os.listdir(job_path) if f[-4:] == ".out" ]

# DOC_PATT = "http://localhost:%d/doc?id=%d&q=%s"

