import caffe
import numpy as np

MODEL_FILE =  "./models/caffe1.prototxt"
PRETRAINED =  "./models/caffemodel1"
caffe_dir = "../caffe"
IMAGE_FILE = "../cat.jpg"
net1 = caffe.Classifier(MODEL_FILE, PRETRAINED,
                   mean=np.load(caffe_dir + '/python/caffe/imagenet/ilsvrc_2012_mean.npy'),
                   channel_swap=(2,1,0),
                   raw_scale=255,
                   image_dims=(256, 256)) 

net1.set_phase_test()
net1.set_mode_gpu()
input_image = caffe.io.load_image(IMAGE_FILE)
out = net1.forward_all(data=np.asarray([net1.preprocess('data', input_image)]))
MODEL_FILE2 =  "./models/caffe2.prototxt"
PRETRAINED2 =  "./models/caffemodel2"
net2 = caffe.Net(MODEL_FILE2, PRETRAINED2)
net2.set_phase_test()
net2.set_mode_gpu()
out2 = net2.forward_all(**out)
with open("synset_words.txt") as f:
    words = f.readlines()
words = map(lambda x: x.strip(), words)
i = out2['prob'][0].argmax(axis=0)
print(i, words[i])

