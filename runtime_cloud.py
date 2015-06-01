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
import psutil
from PIL import Image
from img_util import load_image_from_memory, load_face_from_memory

with open("/home/haichen/datasets/MSRBingFaces/facelabels.txt") as f:
    face_words = f.readlines()[1:]
face_words = map(lambda x: x.strip(), face_words)
with open("/home/haichen/datasets/imagenet/meta/2012/synset_words_caffe.txt") as f:
    words = f.readlines()
words = map(lambda x: x.strip(), words)

model_path = "../../haichen/models/caffe/"

from example import * 
loaded_model = {}
caffe_dir = "../caffe"

class MyTCPHandler(SocketServer.StreamRequestHandler):

    def server_bind(self):
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind(self.server_address)

    def read_n(self, n):
        buf = ''
        while n > 0:
            data = self.rfile.read(n)
            if data == '':
                raise RuntimeError('unexpected connection close')
            buf += data
            n -= len(data)
        return buf

    def load_model(self, target_model, req_type):
        model_file = os.path.join(model_path, target_model+".prototxt")
        pretrained = os.path.join(model_path, target_model+".caffemodel")
        if(req_type == request_pb2.FACE):
            print("loading " + target_model)
            net = face_net([152,152], [152,152], model_file, pretrained, 1)
            return net
        elif(req_type == request_pb2.OBJECT):
            print("loading " + target_model)
            net = caffe.Classifier(model_file, pretrained, mean=np.load(caffe_dir + '/python/caffe/imagenet/ilsvrc_2012_mean.npy'), gpu=True, channel_swap=(2,1,0), image_dims=(256,256), raw_scale=255, batch=1)
            return net
        else:
            print("ERROR")
             

    def handle(self):
        # self.rfile is a file-like object created by the handler;
        # we can now use e.g. readline() instead of raw recv() calls
        global loaded_model
        beg = time.time()
        len_buf = self.rfile.read(4)
        length = struct.unpack('>L', len_buf)[0]
        print("len" + str(length))
        payload = self.read_n(length)
        req = request_pb2.DNNRequest()
        req.ParseFromString(payload)
        target_model = str(req.model)
        if not target_model in loaded_model:
            loaded_model[target_model] = self.load_model(target_model, req.type) 

        model = loaded_model[target_model]

        prob = 0
        latency = 0

        if(req.type == request_pb2.FACE):
            print("starting prediction")
            #input_image = load_face_from_memory(req.data)
            t1 = time.time()
            input_image = load_image_from_memory(req.data) 
            prepared = face_input_prepare(model, [input_image]) 
            out = model.forward_all(**{model.inputs[0]: prepared})
            i = out["prob"].argmax()
            prob = out["prob"].squeeze(axis=(2,3))[0][i]
            label = face_words[i]
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
            images = np.asarray(caffe.io.oversample([input_image], model.crop_dims))
            caffe_in = np.zeros(np.array(images.shape)[[0,3,1,2]],
                             dtype=np.float32)
            for ix, in_ in enumerate(images):
                caffe_in[ix] = model.preprocess('data', in_)
            out = model.forward_all(data=caffe_in)
            prediction = out[model.outputs[0]].squeeze(axis=(2,3))
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
    HOST, PORT = "", 9999 

    # Create the server, binding to localhost on port 9999
    server = SocketServer.TCPServer((HOST, PORT), MyTCPHandler)
    print("SERVER started")

    # Activate the server; this will keep running until you
    # interrupt the program with Ctrl-C
    server.serve_forever()
