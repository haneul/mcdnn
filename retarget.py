import os, sys

# model dir
model_path = "./new_face/"
data_path = "/home/haichen/datasets/MSRBingFaces/"
from example import * 
#lbl = face_recognition(os.path.join(data_path, "iu/112.jpg"), [152,152], [152,152], os.path.join(model_path, "D0.prototxt"), os.path.join(model_path, "D0.caffemodel"))
#net = face_net([152,152], [152,152], os.path.join(model_path, "D0.bottom.prototxt"), os.path.join(model_path, "D0.bottom.caffemodel"))
net = face_net([152,152], [152,152], os.path.join(model_path, "D0.bottom2.prototxt"), os.path.join(model_path, "D0.bottom2.caffemodel"))
input_image = skimage.io.imread(sys.argv[1])
prepared = face_input_prepare(net, [input_image]) 
out = net.forward_all(**{net.inputs[0]: prepared})

#net2 = caffe.Net(os.path.join(model_path, "D0.test.prototxt"), os.path.join(model_path, "face_retarget_train_iter_1000.caffemodel"))
net2 = caffe.Net(os.path.join(model_path, "D0.test2.prototxt"), os.path.join(model_path, "face_retarget2_train_iter_10000.caffemodel"))
#out2 = net2.forward_all(**{net2.inputs[0]: out["fc7"]})
out2 = net2.forward_all(**{net2.inputs[0]: out["Result"]})
print(out2)

print(out2["prob"].argmax())

