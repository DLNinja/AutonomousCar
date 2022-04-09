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

    x = preprocess(frame)
    y = model(x)

    y = F.softmax(y, dim=1)
    
    prob_blocked = float(y.flatten()[0])
    
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

