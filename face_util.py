import os
import caffe
model_path = "face_models"
from example import * 
target1 = ["iu", "barack+obama", "bill+gates", "dr.+dre", "britney+spears", "angelina+jolie", "eminem", "j.k.+rowling", "g-dragon", "dakota+fanning", "bruce+willis", "colin+powell", "seungyeop+han", "matthai+philipose", "jitu+padhye", "haichen+shen", "alec+wolman", "ganesh+ananthanarayanan", "victor+bahl", "peter+bodik", "ratul+mahajan"]
other_words = [["-30", "30-60", "60+"], ["White", "Black", "Hispanic", "S.Asian", "E.Asian", "Other"], ["M", "F"]]


def load_net(option):
    others = []
    targets = ["age", "race", "gender"]

    if option.target == "C0":
        face_net1 = face_net([152,152], [152,152], os.path.join(model_path, "C0.prototxt"), os.path.join(model_path, "C0.caffemodel"), 1, gpu=option.gpu)
        face_net2 = caffe.Net("test_face_c0/test.prototxt", "test_face_c0/face_retarget2_train_iter_8050.caffemodel", 1)
    elif option.target == "D0":
        face_net1 = face_net([152,152], [152,152], os.path.join(model_path, "D0.prototxt"), os.path.join(model_path, "D0.caffemodel"), 1, gpu=option.gpu)
        face_net2 = caffe.Net("test_face/test.prototxt", "test_face/face_retarget2_train_iter_8050.caffemodel", 1)
    else:
        print("no such option")
        raise Exception()
    if option.others:
        for s in targets:
            model = caffe.Net("face_models/%s.%s.prototxt" % (option.target, s), 
                    "face_models/%s.%s.caffemodel" % (option.target, s), 1)
            if not option.sharing:
                fn = face_net([152,152], [152,152], os.path.join(model_path, "%s.prototxt" % option.target), 
                     os.path.join(model_path, "%s.caffemodel" % option.target), 1, gpu=option.gpu)
            else: 
                fn = face_net1
            others.append((fn, model))
    return (face_net1, face_net2, others)
         

def detect_face(input_image, face_net1, face_net2, others, sharing=True):
    prepared = face_input_prepare(face_net1, [input_image]) 
    other_label = [] 
    if len(others) > 0:
        out = face_net1.forward(end="fc7.comput", blobKey="fc7", **{face_net1.inputs[0]: prepared})
        cnt = 0
        for model in others:
            if not sharing:
                out = model[0].forward(end="fc7.comput", blobKey="fc7", **{model[0].inputs[0]: prepared})
            out2 = model[1].forward_all(**{model[1].inputs[0]: out["fc7.comput"]})
            i = out2["prob"].argmax()
            other_label.append(other_words[cnt][i])
            cnt += 1
        
        out = face_net1.forward(start="Result", end="Result", **{face_net1.inputs[0]: prepared})
        out2 = face_net2.forward_all(**{face_net2.inputs[0]: out["Result"]})
    else:
        out = face_net1.forward(end="Result", **{face_net1.inputs[0]: prepared})
        out2 = face_net2.forward_all(**{face_net2.inputs[0]: out["Result"]})
    i = out2["prob"].argmax()
    prob = out2["prob"].squeeze(axis=(2,3))[0][i]
    label = target1[i] + " " + ",".join(other_label)
    return label
