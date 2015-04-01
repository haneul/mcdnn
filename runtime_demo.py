import request_pb2
import struct
import socket
import time
import sys
import numpy as np
import cv2
import time

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
     

cascPath = "opencv_xml/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

cap = cv2.VideoCapture(0)
cap.set(3,320)
cap.set(4,240)
cnt = 0
beg = time.time()
last_face_t = 0
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    now = time.time()
    faces = []
    if (now-last_face_t) > 5:
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor = 1.2,
            minNeighbors = 5,
            minSize=(152, 152),
            flags = cv2.cv.CV_HAAR_SCALE_IMAGE
        )
        #last_face_t = now 
    #print(faces)

    cv2.imshow('frame', frame)
    for x, y, w, h in faces:
        print("Face found!")
        sendFrame(frame[y:y+h, x:x+w], request_pb2.FACE)
    key = cv2.waitKey(1) 
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('c'):
        sendFrame(frame)
    #cnt += 1
    #if cnt == 100: break
end = time.time()
print(cnt/float(end-beg))
cap.release()
cv2.destroyAllWindows()    
