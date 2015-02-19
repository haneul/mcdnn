from caffe.proto import caffe_pb2
import google.protobuf.text_format
import caffe
import numpy as np
param = caffe_pb2.NetParameter()
with open("split/origin.7.Model.prototxt") as f:
    google.protobuf.text_format.Merge(f.read(), param)

    
net2 = caffe.Net("split/origin.7.Model.prototxt", "split/origin.7.Model.caffemodel")
IMAGE_FILE = "../../cat.jpg"
input_image = caffe.io.load_image(IMAGE_FILE)
data=np.asarray([net2.preprocess('data', input_image)])
targets = ["conv1", "conv2", "conv3", "conv4", "conv5", "fc1", "fc2"]
for i in range(len(net2._layer_names)):
    #if(net2._layer_names[i] not in targets): continue
    fname = "split/origin.7.%d.Model.prototxt" % i
    mname = "split/origin.7.%d.Model.caffemodel" % i
    new_param = caffe_pb2.NetParameter()
    for j in range(i, len(net2._layer_names)):
        layer = new_param.layers.add() 
        layer.CopyFrom(param.layers[j])
    new_param.input.append(new_param.layers[0].bottom[0])
    if i == 0:
        new_param.input_dim.extend(param.input_dim)
    else:
        try:
            prediction = net2.forward(data=data, end=net2._layer_names[i-1])
        except:
            continue
        new_param.input_dim.extend(prediction.values()[0].shape)
    with open(fname, "w") as f:
        f.write(google.protobuf.text_format.MessageToString(new_param))
    net3 = caffe.Net(fname, "split/origin.7.Model.caffemodel")
    net3.save(mname)
    del net3
#print(net2._layer_names[i])
#cut = ["pool1","pool2","conv3.comput","conv4.comput","pool5",
#print(param.layers[0])
#print(param.input_dim)

