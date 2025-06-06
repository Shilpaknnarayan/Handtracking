import cv2
import mediapipe as mp
import time
import Handtrackingmodule as htm

pTime = 0
cTime = 0
cap = cv2.VideoCapture(0)
detector=htm.handdetector()
while True:
    sucess, img = cap.read()
    img=detector.findhands(img)
    lmlist=detector.findposition(img,draw=False)
    if len(lmlist)!=0:
        print(lmlist[4])
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
