import sys
#from memory_profiler import profile
#@profile
def main():
    import caffe
    import numpy as np
    caffe_dir = "../caffe"
    MODEL_FILE = caffe_dir + "/models/bvlc_reference_caffenet/deploy.prototxt"
    PRETRAINED = caffe_dir + "/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel"
    IMAGE_FILE = "../cat.jpg"

    with open("synset_words.txt") as f:
        words = f.readlines()
    words = map(lambda x: x.strip(), words)

    net = caffe.Classifier(MODEL_FILE, PRETRAINED,
                           mean=np.load(caffe_dir + '/python/caffe/imagenet/ilsvrc_2012_mean.npy'),
                           channel_swap=(2,1,0),
                           raw_scale=255,
                           image_dims=(256, 256)) 
    caffe.set_phase_test()
    caffe.set_mode_gpu()
    input_image = caffe.io.load_image(IMAGE_FILE)
    #prediction = net.predict([input_image])
    prediction = net.forward_all(data=np.asarray([net.preprocess('data', input_image)]))

    i = prediction["prob"].argmax()
    print(i)
    print(words[i])
#    del net
#    prediction = net2.predict([input_image])
#    i = prediction[0].argmax()
#    print(i)
#    print(words[i])

SETTINGS = {
    "imagenet": {
        "model_file":  "../../haichen/models/caffe/A0.prototxt",
        "pretrained":  "../../haichen/models/caffe/A0.caffemodel",
        "image_dims": (256, 256),
        "input_dims": [224, 224],
        "oversample": False,
    },
    "deepface": {
        "model_file": "tlc/faces/var5.5.Model.prototxt",
        "pretrained": "tlc/faces/var5.5.Model.caffemodel",
        #"model_file": "tlc/faces/faces.var5.rm6.prototxt",
        #"pretrained": "tlc/faces/faces.var5.rm6.caffemodel",
        "image_dims": (152, 152),
        "input_dims": [152, 152],
        "oversample": False,
    },
    "test": {
        "model_file": "test.prototxt",
        "pretrained": "test.caffemodel",
        "image_dims": (3, 3),
        "input_dims": [3, 3],
        "oversample": False,
    }
}

def main_tlc():
    import caffe
    import numpy as np
    import skimage.io
    np.set_printoptions(threshold='nan')

    options = SETTINGS["imagenet"]
    IMAGE_FILE = "../cat.jpg"
    #IMAGE_FILE = "../fish.jpg"
    mean = np.zeros([3] + list(options['input_dims']))
    mean.fill(128.0)
    with open("/home/haichen/datasets/imagenet/meta/2010/synset_words.txt") as f:
        words = f.readlines()
    words = map(lambda x: x.strip(), words)
   
    net = caffe.Classifier(options["model_file"], options["pretrained"],
                           mean=mean, input_scale=0.0078125,
                           image_dims=options["image_dims"])
    sys.stderr.write("model file: %s\n" % options["model_file"])
    sys.stderr.write("pretrained: %s\n" % options["pretrained"])

    caffe.set_phase_test()
    caffe.set_mode_gpu()
    #net.set_mode_cpu()
    """
    with open("rawinput.txt") as f:
        content = f.read()
        rawinput = content.strip(' ,\t\r\n').split(',')
        rawinput = map(lambda x: eval(x), rawinput)
    rawinput = np.asarray(rawinput).reshape([1,3] + options['input_dims'])
    prediction = net.predict_raw(rawinput)
    return
    """

    input_image = skimage.io.imread(IMAGE_FILE)
    prediction = net.predict([input_image], oversample=True)
    #prediction = net.forward_all(data=np.asarray([net.preprocess('data', input_image)]))
    #return
    label = prediction.argmax()
    #for i,v in enumerate(prediction[0]):
    #    print i, v
    print label
    print words[label]
    #print input_image

#main()
main_tlc()

