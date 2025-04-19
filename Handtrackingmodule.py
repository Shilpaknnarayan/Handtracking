import cv2
import mediapipe as mp
import time
class handdetector():
    def __init__(self,mode=False,
               maxhands=2,confidence=0.5,
                 tracking=0.5):
        self.mode=mode
        self.maxhands=maxhands
        self.detection=confidence
        self.tracking=tracking

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxhands,
            min_detection_confidence=self.detection,
            min_tracking_confidence=self.tracking
        )
        self.mpDraw = mp.solutions.drawing_utils
    def findhands(self,img,draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:  # If hands are detected
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return img
    def findposition(self,img,handNo=0,draw=True):
        lmlist=[]
        if self.results.multi_hand_landmarks:
            myhand=self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myhand.landmark):
                print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                #print(id, cx, cy)
                lmlist.append([id,cx,cy])
                if draw:
                    cv2.circle(img,(cx,cy),15,(255,0,255),cv2.FILLED)
        return lmlist
def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector=handdetector()
    while True:
        sucess, img = cap.read()
        img=detector.findhands(img)
        lmlist=detector.findposition(img)
        print(lmlist[4])
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__=="__main__":
    main()
