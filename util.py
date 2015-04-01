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

# detect face
import cv2
import sys
from PIL import Image

cascPath = "opencv_xml/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

#argv[1]:source
#argv[2]:dest
def detect_and_crop(src, dst):
    image = cv2.imread(src)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize=(30, 30),
        flags = cv2.cv.CV_HAAR_SCALE_IMAGE
    )
    cnt = 0
    for face in faces:
        x, y, w, h = face
        box = (x, y, x+w, y+h)
        cropped = image[y:y+h, x:x+w]
        cropped = cv2.resize(cropped, (152,152), interpolation = cv2.INTER_CUBIC)
        cv2.imwrite(dst, cropped)
        cnt += 1

