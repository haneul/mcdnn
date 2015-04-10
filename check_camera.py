
import cv2

cap = cv2.VideoCapture(0)

cap.set(3,640)
cap.set(4,480)
while True:
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    key = cv2.waitKey(1) 
    if key & 0xFF == ord('q'):
        break
