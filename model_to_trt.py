import torch
import torchvision

model = torchvision.models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(512, 2)
model = model.cuda().eval().half()
print("1")
model.load_state_dict(torch.load('best_model_resnet18.pth'))
print("2")
device = torch.device('cuda')
print("3")
from torch2trt import torch2trt

data = torch.zeros((1, 3, 224, 224)).cuda().half()

model_trt = torch2trt(model, [data], fp16_mode=True)
print("4")
torch.save(model_trt.state_dict(), 'best_model_trt.pth')
print("done")
