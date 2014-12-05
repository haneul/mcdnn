import caffe, resource, sys, os, time
import numpy as np

def procStatus():
    pid = os.getpid()
    for line in open("/proc/%d/status" % pid).readlines():
        if line.startswith("VmSize:"):
            return int(line.split(":",1)[1].strip().split(' ')[0])
    return None
r1 = resource.getrusage(resource.RUSAGE_SELF)
net = caffe.Net(sys.argv[1]+".Model.prototxt", sys.argv[1]+".Model.caffemodel")
net.set_phase_test()
net.set_mode_gpu()
del net
m1 = procStatus()
r2 = resource.getrusage(resource.RUSAGE_SELF)
#print (r2.ru_maxrss - r1.ru_maxrss) / 1024.0
beg = time.time()
net2 = caffe.Net(sys.argv[1]+".Model.prototxt", sys.argv[1]+".Model.caffemodel")
end = time.time()
net2.set_phase_test()
net2.set_mode_gpu()
r3 = resource.getrusage(resource.RUSAGE_SELF)
m2 = procStatus()
print "loading time (ms): ",
print (end-beg) * 1000
#print (r3.ru_maxrss - r2.ru_maxrss) / 1024.0
print "memory (MB): ",
print (m2-m1) / 1024.0
IMAGE_FILE = "../cat.jpg"
input_image = caffe.io.load_image(IMAGE_FILE)
data=np.asarray([net2.preprocess('data', input_image)])
for i in range(10):
    prediction = net2.forward_all(data=data)
beg = time.time()
prediction = net2.forward_all(data=data)
end = time.time()
print "compute time (s): ",
print (end-beg) * 1000

