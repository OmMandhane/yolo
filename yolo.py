from ultralytics import YOLO
import cv2
import math
import requests
import cv2
import numpy as np
import imutils
from classess import classNames
import playsound as ps
import threading
import time

model = YOLO('yolov8n.pt')

x = classNames
font = cv2.FONT_HERSHEY_SIMPLEX

# insert your url from the ip webcam app below
url = "http://192.******/shot.jpg"
audioplaying = False
def play(): 
      global audioplaying
      if (audioplaying == False):
       audioplaying = True 
      #  enter the location of your audio file below
       ps.playsound(r"C:\Users\Gayatri\Desktop\om\yolo\audio\HumanDetectedJ.mp3")
       audioplaying = False
      
          
counter = 0
firsttime = True
cap = cv2.VideoCapture(0)
t1 = time.time()
while True:
    img_resp = requests.get(url)
    img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
    img = cv2.imdecode(img_arr, -1)
    img = imutils.resize(img, width=1000, height=1800)
    result = model(img, stream=True)
    for r in result:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1),int(y1),int(x2),int(y2)
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),3)
            conf = math.ceil((box.conf[0]*100))/100
            cls = int(box.cls[0])
            cv2.putText(img,f'{x[cls]}{conf}',(x1,y1),font,1,(0,255,0))  
           
            if cls == 0:
             counter += 1
             if counter == 2:
               t = threading.Thread(target=play)
               t.start()
               counter += 1
             if counter == 12:
                t = threading.Thread(target=play)
                t.start()
                counter = 0 
                        
    cv2.imshow('me',img)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()


