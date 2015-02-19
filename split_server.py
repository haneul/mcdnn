from SimpleXMLRPCServer import SimpleXMLRPCServer
from caffe.proto import caffe_pb2
import google.protobuf.text_format
import xmlrpclib
import caffe
import os
import numpy as np

net = None
input_name = None

def load(layer_id):
    global net, input_name
    fname = "split/origin.7.%d.Model.prototxt" % layer_id
    mname = "split/origin.7.%d.Model.caffemodel" % layer_id 
    param = caffe_pb2.NetParameter()
    if(not os.path.exists(fname)):
        return False
    with open("split/origin.7.%d.Model.prototxt" % layer_id) as f:
        google.protobuf.text_format.Merge(f.read(), param)
    input_name = str(param.input[0])
    net = caffe.Net(fname, mname)
    return True

def predict(data):
    if net == None:
        return -1

    import cStringIO
    inputdata = cStringIO.StringIO(data.data)
    bn = np.load(inputdata)
    #print(input_name)
    #print(bn)
    out2 = net.forward_all(**{input_name:bn})
    i = int(out2['prob'][0].argmax(axis=0)[0][0])
    return i 
    
def check():
    print(net)
    return True

server = SimpleXMLRPCServer(("", 8000))
print "Listening on port 8000..."
server.register_function(load, 'load')
server.register_function(predict, 'predict')
server.register_function(check, 'check')
server.serve_forever()
