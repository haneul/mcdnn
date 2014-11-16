CUDA_DIR := /usr/local/cuda
CAFFE_DIR := /home/ubuntu/caffe

test:
	g++-4.7 -std=c++11 -I /home/ubuntu/caffe/include -I $(CUDA_DIR)/include -c test.cpp -o test.o
	g++-4.7 test.o -L$(CAFFE_DIR)/build/lib -lcaffe -lopencv_core -lopencv_highgui
