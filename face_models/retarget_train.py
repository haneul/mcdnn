import numpy as np
import time
import caffe

beg = time.time()
#traindata = np.load("face.1.train.npy")
#valdata = np.load("face.1.val.npy")
traindata = np.load("train.2.npy")
valdata = np.load("val.2.npy")
end = time.time()

with open("/home/haichen/datasets/MSRBingFaces/labels/msr_cs_faces.n14.90.0.train.shuf.label", "r")  as f:
    train_labels = map(lambda x:int(x.strip().split("\t")[1]), f.readlines())
with open("/home/haichen/datasets/MSRBingFaces/labels/msr_cs_faces.n14.90.0.val.label", "r")  as f:
    val_labels = map(lambda x:int(x.strip().split("\t")[1]), f.readlines())

#solver = caffe.SGDSolver("solver.prototxt")
solver = caffe.SGDSolver("solver2.prototxt")
caffe.set_mode_gpu()
solver.net.set_input_arrays(traindata[:3880], np.array(train_labels[:3880], dtype='f')) 
#print(traindata.shape)
#print(train_labels[:10])
for net in solver.test_nets:
    net.set_input_arrays(valdata[:765],  np.array(val_labels[:765], dtype='f')) 
solver.solve()

