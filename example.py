import caffe
import numpy as np
import skimage.io

# img_file: image filename
# image_dim: dimension of image, [152,152] in this case
# input_dim: dimension of input, [152,152]
def face_recognition(img_file, image_dim, input_dim, model_file, pretrained):
    mean = np.zeros([3] + input_dim)
    scale = 0.00390625
    net = caffe.Classifier(model_file, pretrained, 
        mean=mean, input_scale=scale,
        image_dims=image_dim)
    net.set_phase_test()
    net.set_mode_gpu()

    input_image = skimage.io.imread(img_file)
    prediction = net.predict([input_image], oversample=False)
    label = prediction.argmax()

