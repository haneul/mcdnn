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
name = "test"
faceCount = 0 
lastshot = 0
while True:
    ret, frame = cap.read()
    if ret:
        cv2.imshow('frame', frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        now = time.time()
        if (now-lastshot) > 3:
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
        if(faceCount >= 100): break
    else:
        print("wrong")
    key = cv2.waitKey(1) 
    if key & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()    
