import caffe
import numpy as np
from util import predict

caffe_dir = "../caffe"
MODEL_FILE = caffe_dir + "/models/bvlc_reference_caffenet/deploy.prototxt"
PRETRAINED = caffe_dir + "/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel"
IMAGE_FILE = "../cat.png"
with open("synset_words.txt") as f:
    words = f.readlines()
words = map(lambda x: x.strip(), words)

net = caffe.Classifier(MODEL_FILE, PRETRAINED,
                   mean=np.load(caffe_dir + '/python/caffe/imagenet/ilsvrc_2012_mean.npy'),
                   channel_swap=(2,1,0),
                   raw_scale=255,
                   image_dims=(256, 256)) 
net.set_phase_test()
net.set_mode_gpu()
input_image = caffe.io.load_image(IMAGE_FILE)
#print(list(net._layer_names))
predict([input_image], (256, 256), net.crop_dims, net) 
#r = net.forward_all([input_image])

