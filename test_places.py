import caffe
import numpy as np


with open("/home/haichen/datasets/MITPlaces/trainvalsplit_places205/val_places205.csv", "r")  as f:
    lines = map(lambda x:x.strip(), f.readlines())

MODEL_FILE = "/home/haichen/models/caffe/places205.prototxt"
PRETRAINED = "/home/haichen/models/caffe/places205.caffemodel"
MEAN = "/home/haichen/models/caffe/places205_mean.binaryproto"
with open(MEAN, "rb") as f:
    blob = caffe.proto.caffe_pb2.BlobProto()
    blob.ParseFromString(f.read())
    mean_arr = caffe.io.blobproto_to_array(blob)

net = caffe.Classifier(MODEL_FILE, PRETRAINED, mean=mean_arr[0], gpu=True, channel_swap=(2,1,0), raw_scale=255, image_dims=(256,256))
caffe.set_mode_gpu()
caffe.set_phase_test()

import os

IMAGE_PATH = "/home/haichen/datasets/MITPlaces/vision/torralba/deeplearning/images256"
res = []
lcnt = 0
for i in range(len(lines)/256+1):
    images = []
    for line in lines[i*256:(i+1)*256]:
        sp = line.split()
        path = os.path.join(IMAGE_PATH, sp[0])
        images.append( caffe.io.load_image(path) )
    images = np.asarray(images)
    print(images.shape)
    #prediction = net.forward_all(data=np.asarray(images))
    prediction = net.predict(images, True)
    cnt = 0
    for line in lines[i*256:(i+1)*256]:
        sp = line.strip().split()
        res.append(prediction[cnt].argmax() == int(sp[1]))
        cnt += 1
    lcnt += 1
    print(str(lcnt*256) + "/" + str(len(lines)))
print(sum(res)/float(len(res)))
print(sum(res))
print(len(res))
