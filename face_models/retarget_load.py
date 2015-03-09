import caffe
import sys
import numpy as np
import time
import os
sys.path.append("..")

solver = caffe.SGDSolver("solver2.prototxt")
caffe.set_mode_gpu()
caffe.set_phase_test()

from example import * 
net = face_net([152,152], [152,152], "D0.bottom2.prototxt", "D0.bottom2.caffemodel")

#load train label
#with open("/home/haichen/datasets/MSRBingFaces/labels/msr_cs_faces.n14.90.0.train.shuf.label", "r")  as f:
with open("/home/haichen/datasets/MSRBingFaces/labels/msr_cs_faces.n14.90.0.val.label", "r")  as f:
    lines = map(lambda x:x.strip(), f.readlines())

image_root = "/home/haichen/datasets/MSRBingFaces/images"
images = []
beg = time.time()
for line in lines:
    sp = line.split("\t")
    path = os.path.join(image_root,sp[0])
    images.append(skimage.io.imread(path))
load_end = time.time()
print("loading time: %fs" % (load_end-beg))
#images = images[:10]
prepared = face_input_prepare(net, images)
prepare_end = time.time()
print("preparation time: %fs" % (prepare_end-load_end))
out = net.forward_all(**{net.inputs[0]: prepared})
forward_end = time.time()
print("forward time: %fs" % (forward_end-prepare_end))
np.save("face.val", out["Result"])
