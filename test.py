import caffe

caffe_dir = "../caffe"
MODEL_FILE = caffe_dir + "/models/bvlc_reference_caffenet/deploy.prototxt"
PRETRAINED = caffe_dir + "/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel"
IMAGE_FILE = "images/cat.png"

with open("synset_words.txt") as f:
    words = f.readlines()
words = map(lambda x: x.strip(), words)

import numpy as np
net = caffe.Classifier(MODEL_FILE, PRETRAINED,
                       mean=np.load(caffe_dir + '/python/caffe/imagenet/ilsvrc_2012_mean.npy'),
                       channel_swap=(2,1,0),
                       raw_scale=255,
                       image_dims=(256, 256)) 
net.set_phase_test()
net.set_mode_gpu()

net2 = caffe.Classifier(MODEL_FILE, PRETRAINED,
                       mean=np.load(caffe_dir + '/python/caffe/imagenet/ilsvrc_2012_mean.npy'),
                       channel_swap=(2,1,0),
                       raw_scale=255,
                       image_dims=(256, 256)) 
net2.set_phase_test()
net2.set_mode_gpu()

input_image = caffe.io.load_image(IMAGE_FILE)
prediction = net.predict([input_image])
i = prediction[0].argmax()
print(i)
print(words[i])
del net
prediction = net2.predict([input_image])
i = prediction[0].argmax()
print(i)
print(words[i])


