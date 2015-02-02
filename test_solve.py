import caffe
import sys
import numpy as np
sys.path.append("..")

solver = caffe.SGDSolver("solver.prototxt")
caffe.set_mode_gpu()

from example import * 
net = face_net([152,152], [152,152], "D0.bottom.prototxt", "D0.bottom.caffemodel")

import os
#labels = {"barack+obama":0, "bill+gates":1, "britney+spears":2, "iu":3}
labels = {"arvind+krishnamurthy":0, "haichen+shen":1, "matthai+philipose":2, "seungyeop+han":3}
data_dic = {}
test_dic = {}
#N = 40
N = 5 
#TESTN = 10
TESTN = 1
target_dir = sys.argv[1]

for key in labels:
    lst = filter(lambda x:x.endswith("jpg"), os.listdir(os.path.join(target_dir, key)))[:N]
    data_dic[key] = map(lambda x:os.path.join(target_dir, key, x), lst)
    lst2 = filter(lambda x:x.endswith("jpg"), os.listdir(os.path.join(target_dir, key)))[N:(N+TESTN)]
    test_dic[key] = map(lambda x:os.path.join(target_dir, key, x), lst)

res = []
tests = []
lbls1 = []
lbl_test = []

for key in labels:
    images = []
    for j in data_dic[key]:
        images.append(skimage.io.imread(j))
        lbls1.append(labels[key])
    prepared = face_input_prepare(net, images)
    out = net.forward_all(**{net.inputs[0]: prepared})
    res.append(out["fc7"])

    images = []
    for j in test_dic[key]:
        images.append(skimage.io.imread(j))
        lbl_test.append(labels[key])
    prepared = face_input_prepare(net, images)
    out = net.forward_all(**{net.inputs[0]: prepared})
    tests.append(out["fc7"])

train = np.concatenate((res[0],res[1],res[2],res[3]))
test = np.concatenate((tests[0],tests[1],tests[2],tests[3]))


solver.net.set_input_arrays(train, np.array(lbls1, dtype='f')) 
for net in solver.test_nets:
    net.set_input_arrays(test,  np.array(lbl_test, dtype='f')) 
solver.solve()
print solver.test_nets[0].params['Result'][0].data

