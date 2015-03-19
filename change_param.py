import caffe
import numpy as np
import sys
if(len(sys.argv) < 3): 
    print("$python change_param.py [proto] [pretrained]")
    exit(0)

net = caffe.Net(sys.argv[1], sys.argv[2])
targets = ["fc1"]
for layer_name in targets:
    net.params[layer_name][0].data[0][0][0][0] *= 0.5
net.save(sys.argv[3])
