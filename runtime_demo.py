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

def send_message(sock, message):
    s = message.SerializeToString()
    packed_len = struct.pack('>L', len(s))
    sock.sendall(packed_len + s)

def socket_read_n(sock, n):
    """ Read exactly n bytes from the socket.
        Raise RuntimeError if the connection closed before
        n bytes were read.
    """
    buf = ''
    while n > 0:
        data = sock.recv(n)
        if data == '':
            raise RuntimeError('unexpected connection close')
        buf += data
        n -= len(data)
    return buf


HOST, PORT = "archon.cs.washington.edu", int(sys.argv[1])
# Create a socket (SOCK_STREAM means a TCP socket)
def sendFrame(frame, typ=request_pb2.OBJECT):
    print(frame.shape)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect((HOST, PORT))
        retval, buf = cv2.imencode(".jpg", frame)
        req = request_pb2.DNNRequest()
        req.type = typ 
        req.data = str(bytearray(buf))
        send_message(sock, req)
        len_buf = socket_read_n(sock, 4)
        msg_len = struct.unpack('>L', len_buf)[0]
        msg_buf = socket_read_n(sock, msg_len)
        msg = request_pb2.DNNResponse() 
        msg.ParseFromString(msg_buf)
        end = time.time()
        print(typ, msg.result_str)
    finally:
        sock.close()
    if msg != None:
        return msg.result_str
     

cascPath = "opencv_xml/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
print("capture")
cap = cv2.VideoCapture(0)
print("here?")
print(cap.get(3))
print(cap.get(4))
cap.set(3,640)
cap.set(4,400)
#cap.set(4,480)
cnt = 0
beg = time.time()
last_face_t = 0
puttext_time = 0
lastlabel = ""
label_list = []
print("start reading")
face_mode = False
while True:
    ret, frame = cap.read()
    if not ret:
        continue
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (320, 200), interpolation = cv2.INTER_CUBIC) 
    #gray = cv2.resize(gray, (320, 240), interpolation = cv2.INTER_CUBIC) 
    put = False
    face = False
    now = time.time()
    faces = []
    if face_mode and (now-last_face_t) > 5:
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor = 1.2,
            minNeighbors = 5,
            minSize=(62, 62),
            flags = cv2.cv.CV_HAAR_SCALE_IMAGE
        )
        #last_face_t = now 
    #print(faces)
    #cv2.rectangle(frame, (0, 0), (100,100) + (400, -100), (0,0,255));

    #cv2.putText(frame,"Hello World!!!", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
    cv2.rectangle(frame, (0,0), (640,50), (0,0,0), -1)
    for x, y, w, h in faces:
        print("Face found!")
        x, y, w, h = map(lambda x:2*x, [x,y,w,h]) 
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255)) 
        label = sendFrame(frame[y:y+h, x:x+w], request_pb2.FACE)
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


    cv2.imshow('frame', frame)
    key = cv2.waitKey(1) 
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('c'):
        beg_obj = time.time()
        lastlabel = sendFrame(frame)
        end_obj = time.time()
        print(end_obj-beg_obj)
        puttext_time = time.time()
    elif key & 0xFF == ord('f'):
        face_mode = not face_mode
    elif key & 0xFF == ord('s'):
        lastlabel = sendFrame(frame, request_pb2.SCENE)
        puttext_time = time.time()
    #cnt += 1
    #if cnt == 100: break
end = time.time()
print(cnt/float(end-beg))
cap.release()
cv2.destroyAllWindows()    
