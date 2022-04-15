import cv2
import os
from datetime import datetime
from uuid import uuid1
print(cv2.__version__)

try:
    os.makedirs('Dataset/free')
    os.makedirs('Dataset/blocked')
except FileExistsError:
    print('Directories not created because they already exist')

dispW = 1280
dispH = 720
flip = 0

#nooby 21
camSet='nvarguscamerasrc !  video/x-raw(memory:NVMM), width=3264, height=2464, format=NV12, framerate=21/1 ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'
#cool 120
#camSet='nvarguscamerasrc !  video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=120/1 ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'
#1080p
#camSet='nvarguscamerasrc !  video/x-raw(memory:NVMM), width=1920, height=1080, format=NV12, framerate=30/1 ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'

cam = cv2.VideoCapture(camSet, cv2.CAP_GSTREAMER)

free_counter = len(os.listdir('Dataset/free')) - 1
blocked_counter = len(os.listdir('Dataset/blocked')) - 1

while True:
    ret, frame = cam.read()
    
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('1'):
        free_counter = free_counter + 1
        image_path = os.path.join('Dataset/free', str(len(os.listdir('Dataset/free'))) + '_1.jpg')
        cv2.imwrite(image_path, frame)
    elif key == ord('2'):
        blocked_counter = blocked_counter + 1
        image_path = os.path.join('Dataset/blocked', str(len(os.listdir('Dataset/blocked'))) + '_0.jpg')
        cv2.imwrite(image_path, frame)
    
    font = cv2.FONT_HERSHEY_DUPLEX
    frame = cv2.putText(frame, 'Free: '+str(free_counter), (10, 30), font, 1, (0, 250, 0), 2)
    frame = cv2.putText(frame, 'Blocked: '+str(blocked_counter), (10, 60), font, 1, (0, 0, 250), 2)
    cv2.imshow('nanoCam', frame)
    cv2.moveWindow('nanoCam', 0, 0)

cam.release()
cv2.destroyAllWindows()
