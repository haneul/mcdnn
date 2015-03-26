from caffe.proto import caffe_pb2
import google.protobuf.text_format
import caffe
import numpy as np
#D0-12

target_layer = "lconv6.comput"

path = "/home/haichen/models/caffe/"
IMAGE_FILE = "iu_cropped.jpg"
input_image = caffe.io.load_image(IMAGE_FILE)
for i in range(13):
    param = caffe_pb2.NetParameter()
    param_path = "%sD%i.prototxt" % (path, i)
    pretrained = "%sD%i.caffemodel" % (path, i)
    net = caffe.Net(param_path, pretrained)
    with open(param_path) as f:
        google.protobuf.text_format.Merge(f.read(), param)

    data=np.asarray([net.preprocess('data', input_image)])
    pr = net.forward(data = data, end=target_layer)
    for j in range(len(net._layer_names)):
        if net._layer_names[j] == target_layer:
            break
    print(j) 
    device_param = caffe_pb2.NetParameter()
    for k in range(j+1):
        layer = device_param.layers.add() 
        layer.CopyFrom(param.layers[k])
    device_param.input_dim.extend(param.input_dim)
    device_param.input.extend(param.input)

    targetfname = "new_split/D%d_client.prototxt" % i
    with open(targetfname, "w") as f:
        f.write(google.protobuf.text_format.MessageToString(device_param))

    net3 = caffe.Net(targetfname, pretrained) 
    net3.save("new_split/D%d_client.caffemodel" % i)

    server_param = caffe_pb2.NetParameter()
    for k in range(j+1, len(net._layer_names)):
        layer = server_param.layers.add() 
        layer.CopyFrom(param.layers[k])
    server_param.input_dim.extend(pr.values()[0].shape)
    inp = server_param.input.append(pr.keys()[0])

    targetfname = "new_split/D%d_server.prototxt" % i
    with open(targetfname, "w") as f:
        f.write(google.protobuf.text_format.MessageToString(server_param))

    net3 = caffe.Net(targetfname, pretrained) 
    net3.save("new_split/D%d_server.caffemodel" % i)
