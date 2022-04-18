import cv2
import numpy as np
import time
import smbus

bus = smbus.SMBus(1)
address = 8
data = [0,0]

# functia aceasta este folosita pentru comunicarea dintre jetson nano si arduino
def writeNumber(value):
    bus.write_byte_data(address, 1, value)
    return -1

dispW = 224
dispH = 224
flip = 0
# urmatoarea linie de cod contine diferitele setari ale camerei
camSet='nvarguscamerasrc !  video/x-raw(memory:NVMM), width=3264, height=2464, format=NV12, framerate=10/1 ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'
cam = cv2.VideoCapture(camSet, cv2.CAP_GSTREAMER) # apoi este initializat un obiect care reprezinta camera

while True:
    # prima data, este facuta o poza prin camera platformei
    ret, frame = cam.read()

    if frame is None: 
        continue

    # in urmatoarele linii cream o masca care filtreaza pixelii din imagine care reprezinta culoarea tintei
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lowerPink = np.array([137, 102, 132])
    upperPink = np.array([179, 255, 255])
    
    FGmaskComp = cv2.inRange(hsv, lowerPink, upperPink)

    # functia urmatoare gaseste toate grupurile de pixeli din masca facuta si le introduce intr-un sir
    contours, _ = cv2.findContours(FGmaskComp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True) # apoi acest sir este sortat in functie de ariile grupurilor (aria reprezentand numarul de pixeli)
    if len(contours): # daca sirul contine macar un element inseamna ca acesta este tinta cautata
        cnt = contours[0] # stim ca grupul cu aria cea mai mare este tinta
        area = cv2.contourArea(cnt)
        (x, y, w, h) = cv2.boundingRect(cnt) # cu ajutorul acestei functii sunt luate coordonatele ariei si lungimea si inaltimea acesteia
        if area >= 50: 
            # Cream un crosshair care ne arata unde este tinta
            xc = int(x+w/2) # xc va reprezenta locatia tintei pe axa Ox, iar aceasta valoare va fi trimisa la arduino
            yc = int(y+h/2)
            cv2.line(frame, (0, yc), (xc, yc), (0, 255, 0), 1)
            cv2.line(frame, (xc, yc), (dispW, yc), (0, 255, 0), 1)
            cv2.line(frame, (xc, 0), (xc, yc), (0, 255, 0), 1)
            cv2.line(frame, (xc, yc), (xc, dispH), (0, 255, 0), 1)
            data[1] = xc
            if xc < 112:
                print("Turn left")
            else:
                print("Turn right")
    else:
        print("Target not found, searching...(now ima spin)")
        data[1] = 0
    
    # la final este afisata poza cu liniile create pentru a localiza tinta in imagine
    cv2.imshow('nanoCam', frame)
    cv2.moveWindow('nanoCam', 0, 0)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

    writeNumber(data)
    time.sleep(0.001)

cam.release()
cv2.destroyAllWindows()