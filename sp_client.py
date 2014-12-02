import caffe
import numpy as np
import xmlrpclib

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
proxy = xmlrpclib.ServerProxy("http://localhost:8001/") 
import cStringIO
output = cStringIO.StringIO()
np.save(output, out["pool5"])
#print(proxy.predict(xmlrpclib.Binary(out["pool5"])))
print(proxy.predict(xmlrpclib.Binary(output.getvalue())))

