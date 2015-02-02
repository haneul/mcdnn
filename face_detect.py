# detect face
import cv2
import sys
from PIL import Image

cascPath = "opencv_xml/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
image = cv2.imread(sys.argv[1])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor = 1.2,
    minNeighbors = 5,
    minSize=(30, 30),
    flags = cv2.cv.CV_HAAR_SCALE_IMAGE
)
for face in faces:
    x, y, w, h = face
    box = (x, y, x+w, y+h)
#    orig = Image.open(sys.argv[1])
    cropped = image[y:y+h, x:x+w]
    cropped = cv2.resize(cropped, (152,152), interpolation = cv2.INTER_CUBIC)
    cv2.imwrite(sys.argv[2], cropped)

