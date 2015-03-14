# detect face
import cv2
import sys
from PIL import Image

cascPath = "opencv_xml/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

#argv[1]:source
#argv[2]:dest
import os
src_dir = sys.argv[1]
dst_dir = sys.argv[2]
for fname in os.listdir(src_dir):
    print(fname)
    image = cv2.imread(os.path.join(src_dir, fname))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize=(30, 30),
        flags = cv2.cv.CV_HAAR_SCALE_IMAGE
    )
    cnt = 0
    for face in faces:
        x, y, w, h = face
        box = (x, y, x+w, y+h)
        cropped = image[y:y+h, x:x+w]
        cropped = cv2.resize(cropped, (152,152), interpolation = cv2.INTER_CUBIC)
        cv2.imwrite(os.path.join(dst_dir, "%d%s" %(cnt, fname)), cropped)
        cnt += 1

