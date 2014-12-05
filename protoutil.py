from caffe.proto import caffe_pb2
import google.protobuf.text_format
param = caffe_pb2.NetParameter()
with open("tlc/var2.6.Model.prototxt") as f:
    google.protobuf.text_format.Merge(f.read(), param)
    
#print(param.layers[0])
print(param.input)

