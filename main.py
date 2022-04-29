import cv2
import numpy as np
import time
import smbus

bus = smbus.SMBus(1)
address = 8
data = [0,0]

def writeNumber(value):
    bus.write_byte_data(address, 1, value)
    return -1

dispW = 1280
dispH = 720
flip = 0

camSet='nvarguscamerasrc !  video/x-raw(memory:NVMM), width=3264, height=2464, format=NV12, framerate=21/1 ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'
cam = cv2.VideoCapture(camSet, cv2.CAP_GSTREAMER)

while True:
    ret, frame = cam.read()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    #Pink profile: 137 179 102 255 132 255
    lower = np.array([137, 102, 132])
    upper = np.array([179, 255, 255])

    FGmaskComp = cv2.inRange(hsv, lower, upper)
    cv2.imshow('FG mask comp', FGmaskComp)
    #cv2.moveWindow('FG mask comp', 0, 410)

    contours, _ = cv2.findContours(FGmaskComp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    if len(contours):
        cnt = contours[0]
        area = cv2.contourArea(cnt)
        (x, y, w, h) = cv2.boundingRect(cnt)
        if area >= 10: 
            # Crosshair:
            xc = int(x+w/2)
            yc = int(y+h/2)
            cv2.line(frame, (0, yc), (xc, yc), (0, 255, 0), 1)
            cv2.line(frame, (xc, yc), (dispW, yc), (0, 255, 0), 1)
            cv2.line(frame, (xc, 0), (xc, yc), (0, 255, 0), 1)
            cv2.line(frame, (xc, yc), (xc, dispH), (0, 255, 0), 1)
            if area >= 5000:
                print("Target acquired")
                data[0] = 1
            else:
                print("Go front")
                data[1] = int(xc * 0.19)
                if xc < 320:
                    print("Turn left")
                elif xc>=960:
                    print("Turn right")
            
    else:
        print("Target not found, searching...(now ima spin)")
        data[1] = 0
    
    cv2.imshow('nanoCam', frame)
    #cv2.moveWindow('nanoCam', 0, 0)
    
    #bus.write_i2c_block_data(address, 1, data)
    print(str(data[0])+ " " + str(data[1]))
    time.sleep(0.001)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()


