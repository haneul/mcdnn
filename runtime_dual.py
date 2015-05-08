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
import argparse

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

parser = argparse.ArgumentParser(prog='mcdnn')
parser.add_argument('--cpu', action="store_true", default=False)
parser.add_argument('--nocompact', action="store_false", default=False)
parser.add_argument('--nosharing', action="store_false", default=False)
parser.add_argument('--others', action="store_true", default=False)
args = parser.parse_args()
o = Option()
if args.nocompact:
    o.target = "D0"
else:
    o.target = "C0"

#o.others = args.others 
#o.sharing =  not args.nosharing 
#o.gpu = not args.cpu
#fn1, fn2, others = face_util.load_net(o)
compute_t = 0
ct = 0
network_on = True
HOST, PORT = "archon.cs.washington.edu", 9999 
from util import sendFrame

def check_internet():
    global network_on
    while not network_on:
        print("internet check")
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.connect( (HOST, PORT) )
            network_on = True
            break
        except:
            pass
        time.sleep(3)

import threading
from scheduler import * 

class RuntimeScheduler:
    def __init__(self, name, energy_budget, cost_budget):
        self.name = name
        self.energy_budget = float(energy_budget)
        self.cost_budget = float(cost_budget)
        self.applications = collections.defaultdict(list) 
        self.in_cache = {}
        self.in_cache[Location.DEVICE] = []
        self.in_cache[Location.SERVER] = []
        self.started = time.time()
        
     
    def add_application(self, app_type, application):
        self.applications[app_type].append(application)

    def execute_task(self, app_type, frame):
        tApp = self.applications[app_type][0]
        now = time.time()
        global network_on
        if network_on:
            target_s = tApp.models
        else:
            target_s = []
            
        target_c = tApp.models 
        print(len(target_s), len(target_c))

        if tApp.status == Location.NOTRUNNING: # cold miss
            target_s = filter(lambda x: check_server(x, RTT, latency_limit), target_s) 
            target_c = filter(lambda x: check_device(x, latency_limit), target_c) 

        freqsqsum = sum(map(lambda x:x.freq*x.freq, self.in_cache[Location.SERVER]))
        if tApp.status == Location.SERVER:
            freqsqsum -= (tApp.freq * tApp.freq)
        
        until = 36000
        i = now - self.started 
        target_s = filter(lambda x: check_server_cost(x, self.cost_budget, until-i, server_cost, tApp.freq, freqsqsum), target_s)

        freqsqsum = sum(map(lambda x:x.freq*x.freq, self.in_cache[Location.DEVICE]))
        if tApp.status == Location.DEVICE:
            freqsqsum -= (tApp.freq * tApp.freq)

        target_c = filter(lambda x: check_device_cost(x, self.energy_budget, until-i, tApp.status==Location.SERVER, 
            (i-tApp.last_swapin), tApp.freq, freqsqsum), target_c)

        print(len(target_s), len(target_c))
        #freqsqsum = sum(map(lambda x:x.freq*x.freq, self.in_cache[Location.SERVER]))
        #if tApp.status == Location.SERVER:
        #    freqsqsum -= (tApp.freq * tApp.freq)
        picks = []
        target_s.sort(key=lambda x:x.accuracy, reverse=True)
        target_c.sort(key=lambda x:x.accuracy, reverse=True)
          
        try: 
            server_pick = target_s[0]
            server_pick.location = Location.SERVER
            picks.append(server_pick)
        except:
            server_pick = None
        try:
            client_pick = target_c[0]
            client_pick.location = Location.DEVICE
            picks.append(client_pick)
        except:
            client_pick = None

        print(server_pick)
        print(client_pick)

        picks.sort(key=lambda x:x.accuracy, reverse=True)
        pick = picks[0]
        # update budegt
        if pick.location == Location.SERVER:
            self.cost_budget -= server_pick.s_compute_latency * server_cost
            self.energy_budget -= send_energy

            if tApp.status != Location.SERVER:
                try:
                    self.in_cache[tApp.status].remove(tApp)
                except KeyError:
                    pass
                self.in_cache[pick.location].append(tApp)

        elif pick.location == Location.DEVICE:
            if tApp.status != pick.location:
                self.energy_budget -= client_pick.loading_energy
                tApp.last_swapin = i
                try:
                    self.in_cache[tApp.status].remove(tApp)
                except KeyError:
                    pass
                self.in_cache[pick.location].append(tApp)

            self.energy_budget -= client_pick.compute_energy
        else:
            print("ERROR")
            exit()

        reexecute = False
        if pick.location == Location.SERVER:
            try:
                label, latency = sendFrame(frame, HOST, PORT, request_pb2.FACE, pick.name)
                print(label)
            except:
                network_on = False
                reexecute = True
                t = threading.Timer(3, check_internet)
                t.start()

        if reexecute or pick.location == Location.DEVICE:
            print("local")
            retval, buf = cv2.imencode(".jpg", frame)
            #label = face_util.detect_face(img_util.load_image_from_memory(buf), fn1, fn2, others, o.sharing)

        tApp.status = pick.location
        tApp.pick = pick
        return label


scheduler = RuntimeScheduler("multi", 2*3600, 0.0667)
param = model_pb2.ApplicationModel()
with open("deepface.prototxt") as f:
    google.protobuf.text_format.Merge(f.read(), param)
face_app = Application("deepface", 1, param.models)
scheduler.add_application(AppType.FACE, face_app)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    put = False
    face = False
    now = time.time()
    fps_list.append(now)
    for i in fps_list:
        if now - i > 1:
            fps_list.remove(i)

    faces = []
    if face_mode:# and (now-last_face_t) > 5:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (160, 120), interpolation = cv2.INTER_CUBIC) 
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
        x, y, w, h = map(lambda x:4*x, [x,y,w,h]) 
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255)) 
        t1 = time.time()
        label = scheduler.execute_task(AppType.FACE, frame[y:y+h, x:x+w])
        t2 = time.time()
        compute_t = t2-t1
        put = True
        lastlabel = label
        label_list.append( (now, label) )
        puttext_time = now 
        face = True
        break

    counter = collections.Counter()
    for i in label_list: 
        if now - i[0] > 3:
            label_list.remove(i)
        else:
            counter[i[1]] += 1
    
    if face and len(counter) > 0:
        lastlabel = counter.most_common() [0][0]

    if now - puttext_time < 10:
        cv2.putText(frame,lastlabel, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

    if now - last_fps_update > 1:
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
