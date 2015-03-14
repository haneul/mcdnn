import sys
import caffe

if len(sys.argv) < 3:
    print("python part.py [model] [target proto] [target model]")
    exit(0)
# args: model, target proto, target model
net = caffe.Net(sys.argv[2], sys.argv[1])
net.save(sys.argv[3])
