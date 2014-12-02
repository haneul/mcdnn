from SimpleXMLRPCServer import SimpleXMLRPCServer
import xmlrpclib

import caffe

caffe_dir = "../caffe"

with open("synset_words.txt") as f:
    words = f.readlines()
words = map(lambda x: x.strip(), words)

clients = {}
import numpy as np
import uuid
def register(model_file, pretrained, split=False):
    net = caffe.Classifier(model_file, pretrained,
                           mean=np.load(caffe_dir + '/python/caffe/imagenet/ilsvrc_2012_mean.npy'),
                           channel_swap=(2,1,0),
                           raw_scale=255,
                           image_dims=(256, 256)) 
         
    net.set_phase_test()
    net.set_mode_gpu()
    clientID = str(uuid.uuid1())
    clients[clientID] = (net, split)
    return clientID

def change(clientID, model_file, pretrained, split):
    net = caffe.Classifier(model_file, pretrained,
                           mean=np.load(caffe_dir + '/python/caffe/imagenet/ilsvrc_2012_mean.npy'),
                           channel_swap=(2,1,0),
                           raw_scale=255,
                           image_dims=(256, 256)) 
         
    net.set_phase_test()
    net.set_mode_gpu()
    clients[clientID] = (net, split)
    return clientID
    
def predict(clientID, image_file):
    input_image = caffe.io.load_image(image_file)
    net = clients[clientID][0]
    split = clients[clientID][1]
    prediction = net.forward_all(data=np.asarray([net.preprocess('data', input_image)]))
    if(split):
        import cStringIO
        output = cStringIO.StringIO()
        np.save(output, prediction["pool5"])
        proxy = xmlrpclib.ServerProxy("http://localhost:8001/") 
        r = proxy.predict(xmlrpclib.Binary(output.getvalue()))
        return r
    else:
        i = prediction["prob"].argmax()
        return words[i]

def unregister(clientID):
    del clients[clientID]
    return True
    
#MODEL_FILE = caffe_dir + "/models/bvlc_reference_caffenet/deploy.prototxt"
#PRETRAINED = caffe_dir + "/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel"
#IMAGE_FILE = "images/cat.png"

server = SimpleXMLRPCServer(("localhost", 8000))
print "Listening on port 8000..."
server.register_function(register, 'register')
server.register_function(change, 'change')
server.register_function(predict, 'predict')
server.register_function(unregister, 'unregister')

server.serve_forever()
