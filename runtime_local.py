import request_pb2
import struct
import socket
import time
import sys
import numpy as np
import cv
import cv2
import time
import collections
import face_util
import img_util

cascPath = "opencv_xml/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
print("capture")
cap = cv2.VideoCapture(0)
print("here?")
#print(cap.get(3))
#print(cap.get(4))
cap.set(3,640)
#cap.set(4,400)
cap.set(4,480)
cnt = 0
beg = time.time()
last_face_t = 0
puttext_time = 0
lastlabel = ""
label_list = []
print("start reading")
face_mode = False
fps_list = []
last_fps_update = time.time()
cur_fps = -1

class Option:
    def __init__(self, others=False, sharing=False):
        self.others = others
        self.sharing = sharing 
        self.gpu = True

o = Option()
o.target = "D0"
#o.target = "C0"
o.others = False 
o.sharing = True 
o.gpu = True
fn1, fn2, others = face_util.load_net(o)
compute_t = 0
ct = 0
while True:
    ret, frame = cap.read()
    if not ret:
        continue
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (160, 120), interpolation = cv2.INTER_CUBIC) 
    #gray = cv2.resize(gray, (320, 240), interpolation = cv2.INTER_CUBIC) 
    put = False
    face = False
    now = time.time()
    fps_list.append(now)
    for i in fps_list:
        if now - i > 1:
            fps_list.remove(i)
    faces = []
    if face_mode:# and (now-last_face_t) > 5:
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor = 1.2,
            minNeighbors = 5,
            #minSize=(62, 62),
            minSize=(31, 31),
            flags = cv2.cv.CV_HAAR_SCALE_IMAGE
        )
        #last_face_t = now 
    #print(faces)
    #cv2.rectangle(frame, (0, 0), (100,100) + (400, -100), (0,0,255));

    cv2.rectangle(frame, (0,0), (640,50), (0,0,0), -1)
    t1, t2 = 0,0
    for x, y, w, h in faces:
        #print("Face found!")
        x, y, w, h = map(lambda x:4*x, [x,y,w,h]) 
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255)) 
        retval, buf = cv2.imencode(".jpg", frame[y:y+h, x:x+w])
        t1 = time.time()
        label = face_util.detect_face(img_util.load_image_from_memory(buf), fn1, fn2, others, o.sharing)
        t2 = time.time()
        compute_t = t2-t1
        put = True
        lastlabel = label
        label_list.append( (now, label) )
        puttext_time = now 
        face = True
    counter = collections.Counter()
    for i in label_list: 
        if now - i[0] > 3:
            label_list.remove(i)
        else:
            counter[i[1]] += 1
    
    if face and len(counter) > 0:
        lastlabel = counter.most_common() [0][0]

    if now-puttext_time < 10:
        cv2.putText(frame,lastlabel, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

    if now-last_fps_update > 1:
        last_fps_update = now
        fps = []
        prev = fps_list[0]
        for i in fps_list[1:]:
            fps.append(i-prev)
            prev = i
        if len(fps) > 0:
            cur_fps = len(fps)/float(sum(fps))
        else:
            cur_fps = -1
        ct = compute_t * 1000

    cv2.putText(frame,"fps: %.2f" % cur_fps, (550,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
    cv2.putText(frame,"dnn: %.2fms" % ct, (400,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
    cv2.imshow('frame', frame)
    key = cv2.waitKey(1) 
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('f'):
        face_mode = not face_mode
    #cnt += 1
    #if cnt == 100: break
end = time.time()
print(cnt/float(end-beg))
cap.release()
cv2.destroyAllWindows()    
