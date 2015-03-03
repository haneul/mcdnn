# proto server
import sys
import SocketServer
import struct
import request_pb2
import caffe
import numpy as np
import time
from img_util import load_image_from_memory
caffe_dir = "../caffe"
MODEL_FILE="../../haichen/models/caffe/A0.prototxt"
PRETRAINED="../../haichen/models/caffe/A0.caffemodel"
net = caffe.Classifier(MODEL_FILE, PRETRAINED,
                       mean=np.load(caffe_dir + '/python/caffe/imagenet/ilsvrc_2012_mean.npy'),
                       channel_swap=(2,1,0),
                       raw_scale=255,
                       image_dims=(256, 256)) 
caffe.set_phase_test()
caffe.set_mode_gpu()

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
        payload = self.read_n(length)
        req = request_pb2.DNNRequest()
        req.ParseFromString(payload)
        t1 = time.time()
        """
        with open("test.jpg", "wb") as f:
            f.write(req.data)
        input_image = caffe.io.load_image("test.jpg")
        """
        input_image = load_image_from_memory(req.data)
        t2 = time.time()
        prediction = net.forward_all(data=np.asarray([net.preprocess('data', input_image)]))
        t3 = time.time()
        print((t1-beg)*1000)
        print((t2-beg)*1000)
        print((t3-beg)*1000)

        i = prediction["prob"].argmax()

        print "{} wrote:".format(self.client_address[0])
        response = request_pb2.DNNResponse()
        response.success = True
        response.result = i
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

