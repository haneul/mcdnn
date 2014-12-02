import sys

if(len(sys.argv) < 3): 
    print("$python digest.py [proto] [pretrained]")
    exit(0)

import caffe, hashlib
net = caffe.Net(sys.argv[1], sys.argv[2])
#sys.stdin.readline()
dic = {}
for layer_name in net.params.keys(): 
    h1 = hashlib.sha1(net.params[layer_name][0].data).hexdigest()
    h2 = hashlib.sha1(net.params[layer_name][1].data).hexdigest()
    dic[layer_name] = (h1, h2)
import pickle
with open(sys.argv[3], "w") as f:
    pickle.dump(dic, f)
