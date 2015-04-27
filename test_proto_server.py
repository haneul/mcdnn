# proto server
import sys
import SocketServer
import struct
import request_pb2
import caffe
import numpy as np
import time
import os
import cv2
import sys
from PIL import Image
from img_util import load_image_from_memory, load_face_from_memory
labels = {"arvind+krishnamurthy":0, "haichen+shen":1, "matthai+philipose":2, "seungyeop+han":3}
caffe_dir = "../caffe"
#MODEL_FILE="../../haichen/models/caffe/A0.prototxt"
#PRETRAINED="../../haichen/models/caffe/A0.caffemodel"
MODEL_FILE="../vggnet/VGG_ILSVRC_19_layers.prototxt"
PRETRAINED="../vggnet/VGG_ILSVRC_19_layers.caffemodel"
net = caffe.Classifier(MODEL_FILE, PRETRAINED, 
        mean=np.load(caffe_dir + '/python/caffe/imagenet/ilsvrc_2012_mean.npy'),
        gpu=True, channel_swap=(2,1,0), image_dims=(256,256), raw_scale=255, batch=1)
#with open("synset_words.txt") as f:
#    words = f.readlines()
with open("/home/haichen/datasets/imagenet/meta/2012/synset_words_caffe.txt") as f:
    words = f.readlines()
words = map(lambda x: x.strip(), words)

model_path = "face_models"
data_path = "/home/haichen/datasets/MSRBingFaces/"
from example import * 
#lbl = face_recognition(os.path.join(data_path, "iu/112.jpg"), [152,152], [152,152], os.path.join(model_path, "D0.prototxt"), os.path.join(model_path, "D0.caffemodel"))
#net = face_net([152,152], [152,152], os.path.join(model_path, "D0.bottom.prototxt"), os.path.join(model_path, "D0.bottom.caffemodel"))
#face_net1 = face_net([152,152], [152,152], os.path.join(model_path, "D0.prototxt"), os.path.join(model_path, "D0.caffemodel"), 1)
#face_net2 = caffe.Net("test_face/test.prototxt", "test_face/face_retarget2_train_iter_8050.caffemodel", 1)
face_net1 = face_net([152,152], [152,152], os.path.join(model_path, "C0.prototxt"), os.path.join(model_path, "C0.caffemodel"), 1)
face_net2 = caffe.Net("test_face_c0/test.prototxt", "test_face_c0/face_retarget2_train_iter_8050.caffemodel", 1)
target1 = ["iu", "barack+obama", "bill+gates", "dr.+dre", "britney+spears", "angelina+jolie", "eminem", "j.k.+rowling", "g-dragon", "dakota+fanning", "bruce+willis", "colin+powell", "seungyeop+han", "matthai+philipose", "jitu+padhye", "haichen+shen", "alec+wolman", "ganesh+ananthanarayanan", "victor+bahl", "peter+bodik", "ratul+mahajan", "aakanksha+chowdhery", "arvind+krishnamurthy"]

with open("/home/haichen/datasets/MSRBingFaces/facelabels.txt") as f:
    face_words = f.readlines()[1:]
face_words = map(lambda x: x.strip(), face_words)

MODEL_FILE = "/home/haichen/models/caffe/places205.prototxt"
PRETRAINED = "/home/haichen/models/caffe/places205.caffemodel"
MEAN = "/home/haichen/models/caffe/places205_mean.binaryproto"
with open(MEAN, "rb") as f:
    blob = caffe.proto.caffe_pb2.BlobProto()
    blob.ParseFromString(f.read())
    mean_arr = caffe.io.blobproto_to_array(blob)

with open("scene.txt") as f:
    scene_words = map(lambda x:x.strip(), f.readlines())

scene_net = caffe.Classifier(MODEL_FILE, PRETRAINED, mean=mean_arr[0], gpu=True, channel_swap=(2,1,0), raw_scale=255, image_dims=(256,256))
caffe.set_phase_test()
caffe.set_mode_gpu()

"""
input_image = skimage.io.imread(sys.argv[1])
prepared = face_input_prepare(net, [input_image]) 
out = net.forward_all(**{net.inputs[0]: prepared})
"""

#face_net2 = caffe.Net(os.path.join(model_path, "D0.test2.prototxt"), os.path.join(model_path, "face_retarget2_train_iter_20000.caffemodel"))
#out2 = net2.forward_all(**{net2.inputs[0]: out["Result"]})

import os
import psutil
#p = psutil.Process(os.getpid())
#p.set_cpu_affinity(range(12,24))

class MyTCPHandler(SocketServer.StreamRequestHandler):
    def read_n(self, n):
        buf = ''
        while n > 0:
            data = self.rfile.read(n)
            if data == '':
                raise RuntimeError('unexpected connection close')
            buf += data
            n -= len(data)
        return buf

    def handle(self):
        # self.rfile is a file-like object created by the handler;
        # we can now use e.g. readline() instead of raw recv() calls
        beg = time.time()
        len_buf = self.rfile.read(4)
        length = struct.unpack('>L', len_buf)[0]
        print("len" + str(length))
        payload = self.read_n(length)
        req = request_pb2.DNNRequest()
        req.ParseFromString(payload)
        prob = 0
        latency = 0

        """
        with open("test.jpg", "wb") as f:
            f.write(req.data)
        input_image = caffe.io.load_image("test.jpg")
        """
        if(req.type == request_pb2.FACE):
            print("starting prediction")
            #input_image = load_face_from_memory(req.data)
            t1 = time.time()
            input_image = load_image_from_memory(req.data) 
            prepared = face_input_prepare(face_net1, [input_image]) 
            #out = face_net1.forward_all(**{face_net1.inputs[0]: prepared})
            out = face_net1.forward(end="Result", **{face_net1.inputs[0]: prepared})
            out2 = face_net2.forward_all(**{face_net2.inputs[0]: out["Result"]})
            i = out2["prob"].argmax()
            prob = out2["prob"].squeeze(axis=(2,3))[0][i]
            t2 = time.time()
            latency = t2-t1
            print(prob)
            #label = face_words[i]
            label = target1[i]
            print(i, label)
        elif(req.type == request_pb2.SCENE):
            print("scene")
            input_image = load_image_from_memory(req.data)
            t1 = time.time()
            images = np.asarray(caffe.io.oversample([input_image], scene_net.crop_dims))
            caffe_in = np.zeros(np.array(images.shape)[[0,3,1,2]],
                             dtype=np.float32)
            for ix, in_ in enumerate(images):
                caffe_in[ix] = scene_net.preprocess('data', in_)
            out = scene_net.forward_all(data=caffe_in)
            prediction = out[scene_net.outputs[0]].squeeze(axis=(2,3))
            prediction = prediction.reshape((len(prediction) / 10, 10, -1))
            prediction = prediction.mean(1)
            top5 = prediction.argsort()[0][-5:]
            i = prediction.argmax()
            t2 = time.time()
            latency = t2-t1
            label = scene_words[i]
            top5_label = map(lambda x:scene_words[x].split("\t")[1], top5)
            top5_label.reverse()
            label = ": ".join(top5_label)

            print(i, label)

        else:
        #prediction = net.forward_all(data=np.asarray([net.preprocess('data', input_image)]))
            input_image = load_image_from_memory(req.data)
            t1 = time.time()
            images = np.asarray(caffe.io.oversample([input_image], net.crop_dims))
            caffe_in = np.zeros(np.array(images.shape)[[0,3,1,2]],
                             dtype=np.float32)
            for ix, in_ in enumerate(images):
                caffe_in[ix] = net.preprocess('data', in_)
            out = net.forward_all(data=caffe_in)
            prediction = out[net.outputs[0]].squeeze(axis=(2,3))
            prediction = prediction.reshape((len(prediction) / 10, 10, -1))
            prediction = prediction.mean(1)
            top5 = prediction.argsort()[0][-5:]
            i = prediction.argmax()
            t2 = time.time()
            label = words[i]
            top5_label = map(lambda x:words[x].split(" ",1)[1].split(",")[0], top5)
            top5_label.reverse()
            label = ": ".join(top5_label)
            latency = t2-t1
            print(i, label)

        print "{} wrote:".format(self.client_address[0])
        response = request_pb2.DNNResponse()
        response.success = True
        response.result = i
        response.result_str = label
        response.latency = latency
        #response.confidence = prob
        s = response.SerializeToString()
        packed_len = struct.pack('>L', len(s))
        # Likewise, self.wfile is a file-like object used to write back
        # to the client
        self.wfile.write(packed_len + s)

if __name__ == "__main__":
    #HOST, PORT = "", 9999
    HOST, PORT = "", int(sys.argv[1])

    # Create the server, binding to localhost on port 9999
    server = SocketServer.TCPServer((HOST, PORT), MyTCPHandler)
    print("SERVER started")

    # Activate the server; this will keep running until you
    # interrupt the program with Ctrl-C
    server.serve_forever()

