import caffe, resource, sys, os
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
net2 = caffe.Net(sys.argv[1]+".Model.prototxt", sys.argv[1]+".Model.caffemodel")
net2.set_phase_test()
net2.set_mode_gpu()
r3 = resource.getrusage(resource.RUSAGE_SELF)
m2 = procStatus()
#print (r3.ru_maxrss - r2.ru_maxrss) / 1024.0
print (m2-m1) / 1024.0

