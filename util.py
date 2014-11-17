import numpy as np
import caffe
def predict(inputs, image_dims, crop_dims, net):
    input_ = np.zeros((len(inputs),
        image_dims[0], image_dims[1], inputs[0].shape[2]), 
        dtype=np.float32)
    for ix, in_ in enumerate(inputs):
        input_[ix] = caffe.io.resize_image(in_, image_dims)
    input_ = caffe.io.oversample(input_, crop_dims)
    caffe_in = np.zeros(np.array(input_.shape)[[0,3,1,2]], dtype=np.float32)
    for ix, in_ in enumerate(input_):
        caffe_in[ix] = net.preprocess(net.inputs[0], in_)
    print(net.inputs[0])
    #out = net.forward(**{net.inputs[0]: caffe_in})
    out = net.forward(start="conv1", end="pool5", **{net.inputs[0]: caffe_in})
    print(out["pool5"].shape)
    #out = net.forward(start="fc6")#, end=None)

