import caffe
import sys
import numpy as np
import time
import os
sys.path.append("..")

#solver = caffe.SGDSolver("solver2.prototxt")
#caffe.set_mode_gpu()
#caffe.set_phase_test()

from example import * 
net = face_net([152,152], [152,152], "D0.bottom2.prototxt", "D0.bottom2.caffemodel")
net1 = face_net([152,152], [152,152], "D0.bottom.prototxt", "D0.bottom.caffemodel")



def generate_npy(net, lns, filename, layer_name):
    image_root = "/home/haichen/datasets/MSRBingFaces/images"
    images = []
    beg = time.time()
    for line in lns:
        sp = line.split("\t")
        path = os.path.join(image_root,sp[0])
        images.append(skimage.io.imread(path))
    load_end = time.time()
    print("loading time: %fs" % (load_end-beg))
    prepared = face_input_prepare(net, images)
    prepare_end = time.time()
    print("preparation time: %fs" % (prepare_end-load_end))
    out = net.forward_all(**{net.inputs[0]: prepared})
    forward_end = time.time()
    print("forward time: %fs" % (forward_end-prepare_end))
    np.save(filename, out[layer_name])

#load train label
"""
with open("/home/haichen/datasets/MSRBingFaces/labels/msr_cs_faces.n14.90.0.train.shuf.label", "r")  as f:
    lines = map(lambda x:x.strip(), f.readlines())

generate_npy(net, lines, "train.2", "Result")
generate_npy(net1, lines, "train.1", "fc7")

with open("/home/haichen/datasets/MSRBingFaces/labels/msr_cs_faces.n14.90.0.val.label", "r")  as f:
    lines = map(lambda x:x.strip(), f.readlines())

generate_npy(net, lines, "val.2", "Result")
generate_npy(net1, lines, "val.1", "fc7")

with open("/home/haichen/datasets/MSRBingFaces/labels/msr_cs_faces.n14.90.0.test.label", "r")  as f:
    lines = map(lambda x:x.strip(), f.readlines())

generate_npy(net, lines, "test.2", "Result")
generate_npy(net1, lines, "test.1", "fc7")
"""

with open("/home/haichen/datasets/MSRBingFaces/labels/faces.train.shuf.label", "r")  as f:
    lines = map(lambda x:x.strip(), f.readlines())

generate_npy(net, lines, "D0/train.2", "Result")

