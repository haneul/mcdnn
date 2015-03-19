import sys

if(len(sys.argv) < 3): 
    print("$python digest.py [proto] [pretrained]")
    exit(0)

import caffe, hashlib
import skimage
import numpy as np
net = caffe.Net(sys.argv[1], sys.argv[2])
IMAGE_FILE = "../fish.jpg"
input_image = skimage.io.imread(IMAGE_FILE)
for layer_name in ["conv1", "conv2", "conv3"]: 
    prediction = net.forward(end=layer_name, data=np.asarray([net.preprocess('data', input_image)]))
    print(prediction.values()[0].shape)
#sys.stdin.readline()
#for i in range(len(net.layers)):
#    print(net.layers[i])

