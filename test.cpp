//#include <string>
#include <iostream>
#include "caffe/caffe.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <boost/shared_ptr.hpp> 

using boost::shared_ptr;
using boost::static_pointer_cast;
using std::string;
using caffe::Blob;
using caffe::Caffe;
using caffe::Net;
using caffe::vector;
using caffe::ImageDataLayer;

int main() {
    const string model_path = "models/cppmodel.prototxt";
    string PRETRAINED = "../caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel";
    //string PRETRAINED = "models/caffe_reference_imagenet_model";

    //Caffe::SetDevice(0);
    Caffe::set_mode(Caffe::GPU);
    Caffe::set_phase(Caffe::TEST);

    Net<float> * caffe_net = new Net<float>(model_path);
    caffe_net->CopyTrainedLayersFrom(PRETRAINED);

    //const string image_path = "../cat.png";
    //cv::Mat image = cv::imread(image_path.c_str());
    //vector<cv::Mat> images(1, image);
    //vector<int> labels(1, 0); 
    //const shared_ptr<ImageDataLayer<float> > image_data_layer =
     //       static_pointer_cast<ImageDataLayer<float>>(
      //                      caffe_net->layer_by_name("data"));
    int a; 
    std::cin >> a;

    Net<float> * caffe_net2 = new Net<float>(model_path);
    caffe_net2->CopyTrainedLayersFrom(PRETRAINED);
    std::cin >> a;
    
    Net<float> * caffe_net3 = new Net<float>(model_path);
    caffe_net3->CopyTrainedLayersFrom(PRETRAINED);
    std::cin >> a;
/*    image_data_layer->AddImagesAndLabels(images, labels);
    vector<Blob<float>* > dummy_bottom_vec;
    float loss;
    const vector<Blob<float>*>& result = caffe_net.Forward(dummy_bottom_vec, &loss);
    std::cout << result.size() << std::endl;
    const float* argmaxs = result[1]->gpu_data();
    for (int i = 0; i < result[1]->num(); ++i) {
        LOG(INFO)<< " Image: "<< i << " class:" << argmaxs[i];
    }*/

    delete caffe_net;
    delete caffe_net2;
    return 0;
}
