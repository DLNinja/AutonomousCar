print("Program started")

# cu ajutorul urmatoarelor linii sunt incluse librariile necesare rularii codului
import cv2 # aceasta librarie este necesara pentru folosirea camerei
# urmatoarele librarii sunt necesare pentru utilizarea modelului de machine learning
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import PIL.Image
import numpy as np
import time
import smbus # aceasta librarie este folosita pentru comunicarea i2c intre jetson nano si arduino

print("Imported modules")

bus = smbus.SMBus(1)
address = 8
data = [0,0]

def writeNumber(value):
    bus.write_byte_data(address, 1, value)
    return -1

#Resnet18
#model = torchvision.models.resnet18(pretrained=False)
#model.fc = torch.nn.Linear(512, 2)

#model.load_state_dict(torch.load('best_model_resnet18.pth'))

# prin urmatoarele linii este incarcat modelul de machine learning

#Alexnet:
model = models.alexnet(pretrained=True)
model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, 2)
model.load_state_dict(torch.load('best_model_alexnet_T.pth')) # este incarcat modelul antrenat de noi cu poze

#Squeezenet 1.1
#model = torchvision.models.squeezenet1_1(pretrained=False)
#model.classifier[1] = torch.nn.Conv2d(512, 2, kernel_size=(1,1), stride=(1,1))

#model.load_state_dict(torch.load('best_model_squeeze30.pth'))

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

#cv2.namedWindow('Trackbars')
#cv2.moveWindow('Trackbars', 1320, 0)
#cv2.createTrackbar('hueLower', 'Trackbars', 50, 179, nothing)
#cv2.createTrackbar('hueUpper', 'Trackbars', 100, 179, nothing)
#cv2.createTrackbar('hueLower2', 'Trackbars', 50, 179, nothing)
#cv2.createTrackbar('hueUpper2', 'Trackbars', 100, 179, nothing)
#cv2.createTrackbar('satLower', 'Trackbars', 100, 255, nothing)
#cv2.createTrackbar('satUpper', 'Trackbars', 255, 255, nothing)
#cv2.createTrackbar('valLower', 'Trackbars', 100, 255, nothing)
#cv2.createTrackbar('valUpper', 'Trackbars', 255, 255, nothing)


dispW = 224
dispH = 224
flip = 0
# urmatoarea linie de cod contine diferitele setari ale camerei
camSet='nvarguscamerasrc !  video/x-raw(memory:NVMM), width=3264, height=2464, format=NV12, framerate=10/1 ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'
cam = cv2.VideoCapture(camSet, cv2.CAP_GSTREAMER) # apoi este initializat un obiect care reprezinta camera

print("Live demo starts now...")

while True:
    # prima data, este facuta o poza prin camera din fata platformei
    ret, frame = cam.read()

    if frame is None:
        continue
    
    since = time.time()
    # poza este procesata pentru a putea fi folosita de modelul de machine learning
    x = preprocess(frame)
    y = model(x)

    y = F.softmax(y, dim=1)
    
    prob_blocked = float(y.flatten()[0])

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    #Pink profile: 137 179 102 255 132 255
    lowerPink = np.array([137, 102, 132])
    upperPink = np.array([179, 255, 255])
    
    #hueLow = cv2.getTrackbarPos('hueLower', 'Trackbars')
    #hueUp = cv2.getTrackbarPos('hueUpper', 'Trackbars')

    #hueLow2 = cv2.getTrackbarPos('hueLower2', 'Trackbars')
    #hueUp2 = cv2.getTrackbarPos('hueUpper2', 'Trackbars')

    #Ls = cv2.getTrackbarPos('satLower', 'Trackbars')
    #Us = cv2.getTrackbarPos('satUpper', 'Trackbars')

    #Lv = cv2.getTrackbarPos('valLower', 'Trackbars')
    #Uv = cv2.getTrackbarPos('valUpper', 'Trackbars')
    
    
    #lowerB = np.array([hueLow, Ls, Lv])
    #upperB = np.array([hueUp, Us, Uv])
    #lowerB2 = np.array([hueLow2, Ls, Lv])
    #upperB2 = np.array([hueUp2, Us, Uv])

    #FGmask = cv2.inRange(hsv, lowerB, upperB)
    #FGmask2 = cv2.inRange(hsv, lowerB2, upperB2)
    #FGmaskComp = cv2.add(FGmask, FGmask2)
    FGmaskComp = cv2.inRange(hsv, lowerPink, upperPink)
    #cv2.imshow('FG mask comp', FGmaskComp)
    #cv2.moveWindow('FG mask comp', 0, 410)

    contours, _ = cv2.findContours(FGmaskComp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    if len(contours):
        cnt = contours[0]
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
            #print("Target found, coordinates: "+str(xc)+":"+str(yc))
            if prob_blocked < 0.5:
                print("Go front")
                data[1] = 0
           else:
                data[1] = xc
                if xc < 112:
                    
                    print("Turn left")
                else:
                    print("Turn right")
    else:
        print("Target not found, searching...(now ima spin)")
        data[1] = 0
    font = cv2.FONT_HERSHEY_DUPLEX
    frame = cv2.putText(frame, 'Free: '+str(float(1-prob_blocked)), (10, 30), font, 1, (0, 250, 0), 1)
    frame = cv2.putText(frame, 'Blocked: '+str(float(prob_blocked)), (10, 60), font, 1, (0, 0, 250), 1)
    
    cv2.imshow('nanoCam', frame)
    cv2.moveWindow('nanoCam', 0, 0)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    #print(time.time()-since)
    time.sleep(0.001)
    writeNumber(data)
cam.release()
cv2.destroyAllWindows()

