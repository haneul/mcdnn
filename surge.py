import caffe
import numpy as np

caffe_dir = "../caffe"
MODEL_FILE =  "./models/caffe1.prototxt"
PRETRAINED = caffe_dir + "/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel"
net = caffe.Net(MODEL_FILE, PRETRAINED)
net.save('models/caffemodel1')
MODEL_FILE =  "./models/caffe2.prototxt"
net = caffe.Net(MODEL_FILE, PRETRAINED)
net.save('models/caffemodel2')
