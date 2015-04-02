#!/usr/bin/python
import os, sys

# model dir
#model_path = "/home/haichen/models/caffe/"
model_path = "face_models/"
data_path = "/home/haichen/datasets/MSRBingFaces/"
from example import * 
#lbl = face_recognition(os.path.join(data_path, "iu/112.jpg"), [152,152], [152,152], os.path.join(model_path, "D0.prototxt"), os.path.join(model_path, "D0.caffemodel"))
#lbl = face_recognition(sys.argv[1], [152,152], [152,152], os.path.join(model_path, "D0.prototxt"), os.path.join(model_path, "D0.caffemodel"))
net = face_net([152,152], [152,152], os.path.join(model_path, "D0.prototxt"), os.path.join(model_path, "D0.caffemodel"))
caffe.set_mode_gpu()
target = "iu_cropped.jpg"
if len(sys.argv) > 1:
    target = sys.argv[1]
#print(lbl)
input_image = skimage.io.imread(target)
prepared = face_input_prepare(net, [input_image])
for i in range(10):
    out = net.forward_all(**{net.inputs[0]: prepared})
print(out["prob"].argmax())

