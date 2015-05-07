import numpy as np
import caffe
import request_pb2
import socket
import struct
import time
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

def send_message(sock, message):
    s = message.SerializeToString()
    packed_len = struct.pack('>L', len(s))
    sock.sendall(packed_len + s)

def socket_read_n(sock, n):
    """ Read exactly n bytes from the socket.
        Raise RuntimeError if the connection closed before
        n bytes were read.
    """
    buf = ''
    while n > 0:
        data = sock.recv(n)
        if data == '':
            raise RuntimeError('unexpected connection close')
        buf += data
        n -= len(data)
    return buf

# Create a socket (SOCK_STREAM means a TCP socket)
def sendFrame(frame, HOST, PORT, typ=request_pb2.OBJECT):
    print(frame.shape)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect((HOST, PORT))
        retval, buf = cv2.imencode(".jpg", frame)
        req = request_pb2.DNNRequest()
        req.type = typ 
        req.data = str(bytearray(buf))
        send_message(sock, req)
        len_buf = socket_read_n(sock, 4)
        msg_len = struct.unpack('>L', len_buf)[0]
        msg_buf = socket_read_n(sock, msg_len)
        msg = request_pb2.DNNResponse() 
        msg.ParseFromString(msg_buf)
        end = time.time()
        print(typ, msg.result_str)
    finally:
        sock.close()
    if msg != None:
        return msg.result_str, msg.latency*1000


def specialize(lst):
    beg = time.time() 
    traindata = np.load("face_models/C0/train.2.npy")
    valdata = np.load("face_models/C0/test.2.npy")
    trainlabel = np.load("face_models/train.label.npy")
    vallabel = np.load("face_models/test.label.npy")
    indices = []
    for i in range(len(trainlabel)):
        if(trainlabel[i] in lst):
            indices.append(i)
    traindata = traindata[indices,:,:,:]
    trainlabel = np.array(trainlabel[indices],dtype="f")
    indices = []
    for i in range(len(vallabel)):
        if(vallabel[i] in lst):
            indices.append(i)
    valdata = valdata[indices,:,:,:]
    vallabel = vallabel[indices]

    solver = caffe.SGDSolver("c0_solver2.prototxt")
    caffe.set_mode_gpu()
    solver.net.set_input_arrays(traindata, trainlabel)
    for net in solver.test_nets:
        net.set_input_arrays(valdata,  np.array(vallabel, dtype='f')) 
    solver.solve()
    end = time.time()
    print(end-beg)

