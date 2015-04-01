import cv2
import skimage.io
import numpy as np
from PIL import Image

def load_image_from_memory(data, color=True):
    img = cv2.imdecode(np.asarray(bytearray(data)), cv2.CV_LOAD_IMAGE_COLOR)    
    b,g,r = cv2.split(img)
    img = cv2.merge((r,g,b))
    img = skimage.img_as_float(img).astype(np.float32)
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
        if color:
            img = np.tile(img, (1, 1, 3))
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    return img

cascPath = "opencv_xml/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

def load_face_from_memory(data):
    img = cv2.imdecode(np.asarray(bytearray(data)), cv2.CV_LOAD_IMAGE_COLOR)    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize=(30, 30),
        flags = cv2.cv.CV_HAAR_SCALE_IMAGE
    )
    if len(faces) == 0:
        return
    for face in faces:
        x, y, w, h = face
        box = (x, y, x+w, y+h)
        cropped = img[y:y+h, x:x+w]
        cropped = cv2.resize(cropped, (152,152), interpolation = cv2.INTER_CUBIC)
        break
    b,g,r = cv2.split(cropped)
    img = cv2.merge((r,g,b))
    img = skimage.img_as_float(img).astype(np.float32)
    return img


    
