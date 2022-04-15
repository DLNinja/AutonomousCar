print("it started")
import cv2
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import PIL.Image
import numpy as np
import time

print("Imported modules")

model = torchvision.models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(512, 2)

model.load_state_dict(torch.load('best_model_resnet18.pth'))

device = torch.device('cuda')
model = model.to(device)
model = model.eval().half()

print("Loaded the model")

mean = torch.Tensor([0.485, 0.456, 0.406]).cuda().half()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda().half()

normalize = torchvision.transforms.Normalize(mean, std)

def preprocess(image):
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device).half()
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]

def nothing(x):
    pass

cv2.namedWindow('Trackbars')
#cv2.moveWindow('Trackbars', 1320, 0)
cv2.createTrackbar('hueLower', 'Trackbars', 50, 179, nothing)
cv2.createTrackbar('hueUpper', 'Trackbars', 100, 179, nothing)
cv2.createTrackbar('hueLower2', 'Trackbars', 50, 179, nothing)
cv2.createTrackbar('hueUpper2', 'Trackbars', 100, 179, nothing)
cv2.createTrackbar('satLower', 'Trackbars', 100, 255, nothing)
cv2.createTrackbar('satUpper', 'Trackbars', 255, 255, nothing)
cv2.createTrackbar('valLower', 'Trackbars', 100, 255, nothing)
cv2.createTrackbar('valUpper', 'Trackbars', 255, 255, nothing)


dispW = 224
dispH = 224
flip = 2
#nooby 21
camSet='nvarguscamerasrc !  video/x-raw(memory:NVMM), width=3264, height=2464, format=NV12, framerate=21/1 ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'
#cool 120
#camSet='nvarguscamerasrc !  video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=120/1 ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'
#1080p
#camSet='nvarguscamerasrc !  video/x-raw(memory:NVMM), width=1920, height=1080, format=NV12, framerate=30/1 ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'

cam = cv2.VideoCapture(camSet, cv2.CAP_GSTREAMER)
print("Live demo starts now...")
while True:
    ret, frame = cam.read()

    if frame is None:
        continue

    x = preprocess(frame)

    y = model(x)

    y = F.softmax(y, dim=1)
    
    prob_blocked = float(y.flatten()[0])

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    hueLow = cv2.getTrackbarPos('hueLower', 'Trackbars')
    hueUp = cv2.getTrackbarPos('hueUpper', 'Trackbars')

    hueLow2 = cv2.getTrackbarPos('hueLower2', 'Trackbars')
    hueUp2 = cv2.getTrackbarPos('hueUpper2', 'Trackbars')

    Ls = cv2.getTrackbarPos('satLower', 'Trackbars')
    Us = cv2.getTrackbarPos('satUpper', 'Trackbars')

    Lv = cv2.getTrackbarPos('valLower', 'Trackbars')
    Uv = cv2.getTrackbarPos('valUpper', 'Trackbars')
    
    lowerB = np.array([hueLow, Ls, Lv])
    upperB = np.array([hueUp, Us, Uv])
    lowerB2 = np.array([hueLow2, Ls, Lv])
    upperB2 = np.array([hueUp2, Us, Uv])

    FGmask = cv2.inRange(hsv, lowerB, upperB)
    FGmask2 = cv2.inRange(hsv, lowerB2, upperB2)
    FGmaskComp = cv2.add(FGmask, FGmask2)
    cv2.imshow('FG mask comp', FGmaskComp)
    cv2.moveWindow('FG mask comp', 0, 410)

    contours, _ = cv2.findContours(FGmaskComp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        (x, y, w, h) = cv2.boundingRect(cnt)
        if area >= 50:
            # Crosshair:
            xc = int(x+w/2)
            yc = int(y+h/2)
            cv2.line(frame, (0, yc), (xc, yc), (0, 255, 0), 1)
            cv2.line(frame, (xc, yc), (dispW, yc), (0, 255, 0), 1)
            cv2.line(frame, (xc, 0), (xc, yc), (0, 255, 0), 1)
            cv2.line(frame, (xc, yc), (xc, dispH), (0, 255, 0), 1)
    
    font = cv2.FONT_HERSHEY_DUPLEX
    frame = cv2.putText(frame, 'Free: '+str(float(1-prob_blocked)), (10, 30), font, 1, (0, 250, 0), 1)
    frame = cv2.putText(frame, 'Blocked: '+str(float(prob_blocked)), (10, 60), font, 1, (0, 0, 250), 1)
    
    cv2.imshow('nanoCam', frame)
    cv2.moveWindow('nanoCam', 0, 0)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    time.sleep(0.001)

cam.release()
cv2.destroyAllWindows()

