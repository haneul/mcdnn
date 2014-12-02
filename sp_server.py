from SimpleXMLRPCServer import SimpleXMLRPCServer
import xmlrpclib

import caffe

caffe_dir = "../caffe"

with open("synset_words.txt") as f:
    words = f.readlines()
words = map(lambda x: x.strip(), words)

MODEL_FILE2 =  "./models/caffe2.prototxt"
PRETRAINED2 =  "./models/caffemodel2"
net2 = caffe.Net(MODEL_FILE2, PRETRAINED2)
net2.set_phase_test()
net2.set_mode_gpu()
#out2 = net2.forward_all(**out)

clients = {}
import numpy as np
import uuid
    
def predict(data):
    import cStringIO
    inputdata = cStringIO.StringIO(data.data)
    bn = np.load(inputdata)
    out2 = net2.forward_all(**{"pool5":bn})
    i = out2['prob'][0].argmax(axis=0)
    return words[i]

#MODEL_FILE = caffe_dir + "/models/bvlc_reference_caffenet/deploy.prototxt"
#PRETRAINED = caffe_dir + "/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel"
#IMAGE_FILE = "images/cat.png"

server = SimpleXMLRPCServer(("localhost", 8001))
print "Listening on port 8001..."
server.register_function(predict, 'predict')

server.serve_forever()
