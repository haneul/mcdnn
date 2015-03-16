import log_pb2

memories = {}

# load log into memory
def load(log_file):
    logs = log_pb2.RMLogs()
    with open(log_file, "rb") as f:
        logs.ParseFromString(f.read())
    start = -1
    memory_entries = filter(lambda x:x.type == log_pb2.MEMORY, logs.entry)
    for entry in memory_entries:
        if start == -1:
            start = entry.datetime
        t = entry.datetime - start
        memories[t] = entry.data
import sys
load(sys.argv[1])
keys = memories.keys()
keys.sort()

import bisect

TOTAL_MEM = 2*1024*1024*1024
def available_mem(t):
    l = bisect.bisect_left(keys, t) 
    return TOTAL_MEM - memories[keys[l-1]]

print(available_mem(1000))
