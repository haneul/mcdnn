import cv2
import time

cascPath = "opencv_xml/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,400)
target = "faces/"
#name = "matthai+philipose/"
#name = "seungyeop+han/"
name = "test/"
faceCount = 0 
lastshot = 0
turn = 0
collectFace = False
lastTurn = 0  
while True:
    ret, frame = cap.read()
    #if faceCount % 20 == 0 and lastTurn != faceCount:
    #    raw_input("turn %d" % turn)
    #    lastTurn = faceCount
    #    turn += 1

    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        now = time.time()
        faces = []
        if collectFace and (now-lastshot) > 0.2:
            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor = 1.2,
                minNeighbors = 5,
                minSize=(152, 152),
                flags = cv2.cv.CV_HAAR_SCALE_IMAGE
            )
        for x, y, w, h in faces:
            print("Face found! %d" % faceCount)
            face = frame[y:y+h, x:x+w]
            face = cv2.resize(face, (152, 152), interpolation = cv2.INTER_CUBIC)
            cv2.imwrite("%s%s%d.jpg" % (target, name, faceCount), face) 
            faceCount += 1
            lastshot = now
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255)) 
        cv2.imshow('frame', frame)
        if collectFace and (faceCount % 20 == 0) and lastTurn != faceCount:
            collectFace = False
            lastTurn = faceCount
            print("paused %d" % faceCount)
        if(faceCount >= 100): break
    else:
        print("wrong")
    key = cv2.waitKey(1) 
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('f'):
        print("restarted %d" % turn)
        turn += 1
        collectFace = True

cap.release()
cv2.destroyAllWindows()    
