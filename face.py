import os, sys

# model dir
model_path = "/home/haichen/models/caffe/"
data_path = "/home/haichen/datasets/MSRBingFaces/"
from example import face_recognition 
#lbl = face_recognition(os.path.join(data_path, "iu/112.jpg"), [152,152], [152,152], os.path.join(model_path, "D0.prototxt"), os.path.join(model_path, "D0.caffemodel"))
lbl = face_recognition(sys.argv[1], [152,152], [152,152], os.path.join(model_path, "D0.prototxt"), os.path.join(model_path, "D0.caffemodel"))
print(lbl)

