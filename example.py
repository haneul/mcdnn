import caffe
import numpy as np
import skimage.io

def face_net(image_dim, input_dim, model_file, pretrained, batch=10, gpu=True):
    mean = np.zeros([3] + input_dim)
    scale = 0.00390625
    net = caffe.Classifier(model_file, pretrained, 
        mean=mean, input_scale=scale,
        #image_dims=image_dim, batch=batch)
        image_dims=image_dim, batch=batch, gpu=gpu)
    #caffe.set_phase_test()
    #net.set_mode_gpu()
    return net

def face_input_prepare(net, inputs, oversample=False):
    # Scale to standardize input dimensions.
    input_ = np.zeros((len(inputs),
        net.image_dims[0], net.image_dims[1], inputs[0].shape[2]),
        dtype=np.float32)
    for ix, in_ in enumerate(inputs):
        #input_[ix] = caffe.io.resize_image(in_, net.image_dims)
        input_[ix] = caffe.io.resize_image_tlc(in_, net.image_dims)

    if oversample:
        # Generate center, corner, and mirrored crops.
        input_ = caffe.io.oversample(input_, net.crop_dims)
    else:
        # Take center crop.
        center = np.array(net.image_dims) / 2.0 
        crop = np.tile(center, (1, 2))[0] + np.concatenate([
            -net.crop_dims / 2.0,
            net.crop_dims / 2.0 
        ])  
        input_ = input_[:, crop[0]:crop[2], crop[1]:crop[3], :]

    # Classify
    caffe_in = np.zeros(np.array(input_.shape)[[0,3,1,2]],
                        dtype=np.float32)
    for ix, in_ in enumerate(input_):
        caffe_in[ix] = net.preprocess(net.inputs[0], in_)

    return caffe_in

# img_file: image filename
# image_dim: dimension of image, [152,152] in this case
# input_dim: dimension of input, [152,152]
def face_recognition(img_file, image_dim, input_dim, model_file, pretrained):
    mean = np.zeros([3] + input_dim)
    scale = 0.00390625
    net = caffe.Classifier(model_file, pretrained, 
        mean=mean, input_scale=scale,
        image_dims=image_dim)
    #net.set_phase_test()
    #net.set_mode_gpu()

    input_image = skimage.io.imread(img_file)
    prediction = net.predict([input_image], oversample=False)
    label = prediction.argmax()
    return label
