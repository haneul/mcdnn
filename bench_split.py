import xmlrpclib
import caffe
import numpy as np
import time

proxy = xmlrpclib.ServerProxy("http://archon.cs.washington.edu:8000/")
import cStringIO

net = caffe.Net("memory/origin.7.Model.prototxt", "memory/origin.7.Model.caffemodel")
net.set_phase_test()
net.set_mode_gpu()

IMAGE_FILE = "../cat.jpg"
input_image = caffe.io.load_image(IMAGE_FILE)
for i in range(10):
    out = net.forward_all(data=np.asarray([net.preprocess('data', input_image)]))

for i in range(21):
    if(not proxy.load(i)): continue
    d1 = np.asarray([net.preprocess('data', input_image)])
    output = cStringIO.StringIO()
    if(i == 0):
        np.save(output, d1)
        beg = time.time()
        proxy.predict(xmlrpclib.Binary(output.getvalue()))
        end = time.time()
        print("0\t0\t" + str(d1.size) + "\t" + str((end-beg)*1000))
    else:
        data = np.asarray([net.preprocess('data', input_image)])
        for j in range(10):
            out = net.forward(data=data, end=net._layer_names[i-1])
            np.save(output, out.values()[0])
            #proxy.predict(xmlrpclib.Binary(output.getvalue()))
        beg = time.time()
        out = net.forward(data=data, end=net._layer_names[i-1])
        mid = time.time()
        np.save(output, out.values()[0])
        proxy.predict(xmlrpclib.Binary(output.getvalue()))
        end = time.time()
        print(str(i) + "\t" + str((mid-beg)*1000) + "\t" + str(out.values()[0].size) + "\t" + str((end-mid)*1000))
