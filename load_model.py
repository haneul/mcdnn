import model_pb2
import google.protobuf.text_format

param = model_pb2.ApplicationModel()
with open("model_sample.prototxt") as f:
    google.protobuf.text_format.Merge(f.read(), param)

for model in param.models:
    print(model)
    break

