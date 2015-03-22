import caffe
import numpy as np


#with open("/home/haichen/datasets/MITPlaces/trainvalsplit_places205/val_places205.csv", "r")  as f:
with open("/home/haichen/datasets/imagenet/meta/2012/val_caffe.txt", "r")  as f:
    lines = map(lambda x:x.strip(), f.readlines())


MODEL_FILE = "/home/syhan/vggnet/VGG_ILSVRC_19_layers.prototxt"
PRETRAINED = "/home/syhan/vggnet/VGG_ILSVRC_19_layers.caffemodel"

caffe_dir = "../caffe"
net = caffe.Classifier(MODEL_FILE, PRETRAINED, 
        mean=np.load(caffe_dir + '/python/caffe/imagenet/ilsvrc_2012_mean.npy'),
        gpu=True, channel_swap=(2,1,0), image_dims=(256,256), raw_scale=255, batch=32)
        #gpu=True, channel_swap=(2,1,0), raw_scale=255, image_dims=(256,256))
caffe.set_mode_gpu()
caffe.set_phase_test()

import os

IMAGE_PATH = "/home/haichen/datasets/imagenet/ILSVRC2012/val"
res = []
lcnt = 0
BATCH = 256
rr = []
#BATCH = 256
for i in range(len(lines)/BATCH+1):
    images = []
    for line in lines[i*BATCH:(i+1)*BATCH]:
        sp = line.split()
        path = os.path.join(IMAGE_PATH, sp[0])
        images.append( caffe.io.load_image(path) )
    images = np.asarray(images)
    print(images.shape)
    #prediction = net.forward_all(data=np.asarray(images))
    prediction = net.predict(images, True)
    cnt = 0
    for line in lines[i*BATCH:(i+1)*BATCH]:
        sp = line.strip().split()
        rr.append( (prediction[cnt].argmax(), sp[1]) )
        res.append(prediction[cnt].argmax() == int(sp[1]))
        cnt += 1
    lcnt += 1
    print(str(lcnt*BATCH) + "/" + str(len(lines)))
print(rr)
print(sum(res)/float(len(res)))
print(sum(res))
print(len(res))
