from __future__ import print_function,division
import torch
from torch import optim,nn,device
from torch.utils.data import DataLoader
import numpy as np
import torchvision
import torch.nn.functional as F
from torchvision.models import resnet50,AlexNet,resnet18
from torch.utils.data import TensorDataset
from torchvision import datasets,models,transforms
import time,os,copy
import cv2
from dataset import FaceClassDataset
from flask import Flask, request, jsonify, render_template
import io,base64
from network import myModle_class,LoadCNN
from PIL import Image 

app = Flask(__name__)
app.config['JSONIFY_MIMETYPE'] ="application/json;charset=utf-8"   # 指定浏览器渲染的文件类型，和解码格式；
device = torch.device('cuda' if not torch.cuda.is_available() else 'cpu')


faceClasses = 7
#names = ['唐僧','姚明','张学友','曹寅','郭爱斌','饶志豪', '曹焕琪']
names = ['tangseng','yaoming','zhangxueyou','caoyin','guoaibin','raozhihao','caohuanqi']


#model = myModle_class(faceClasses)
model = LoadCNN(faceClasses)
#model = torch.load('./class_face.pth')
model.load_state_dict(torch.load('./class_face.pth'))
model.to(device)


data_transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.RandomRotation(20),
    transforms.ToTensor()
])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    ################## 构建数据集  ##########################

    
    data = request.get_json(force=True)
    facePath = data['facePath']
    img = Image.open(facePath).convert("RGB")
    inputs = data_transform(img)
    inputs = torch.unsqueeze(inputs,0)
    ############  开始评估  #################################
   
    model.eval()
    inputs = inputs.to(device)
    out = model(inputs)
    F.softmax(out,)
    index = torch.argmax(out,1)
    name = names[index.cpu()[0].item()]
    #return name
    return jsonify({'result': name})

@app.route('/predict2', methods=['POST'])
def predict2():
    data = request.get_json(force=True)
    base64_decoded = base64.b64decode(data['imageData'])
    
    img = Image.open(io.BytesIO(base64_decoded)).convert("RGB")
    inputs = data_transform(img)
    inputs = torch.unsqueeze(inputs,0)

    model.eval()
    inputs = inputs.to(device)
    out = model(inputs)
    res = F.softmax(out[0])
    index = torch.argmax(res,0)
    if res[index.item()] < 0.5:
        name = 'error'
    else:
        name = names[index.item()]
    
    #return name
    return jsonify({'result': name})

def main():
    ################# 加载模型  ####################################
    model_path = "./unlock_class/unlock.pth"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = myModle(num_classes=2)  
    if os.path.exists(model_path):
        model = torch.load(model_path) 
    model.to(device)

    ################## 构建数据集  ##########################
    data_transform = transforms.Compose([
        transforms.Resize((128,128)),
        # transforms.RandomRotation(20),
        # transforms.RandomVerticalFlip(),
        transforms.RandomRotation(40),
        transforms.ToTensor()
    ])
    
    train_set = datasets.ImageFolder("X:/Unlock_dataset/train",data_transform)
    # test_set = datasets.ImageFolder("X:/Unlock_dataset/19unlock1",data_transform)
    test_set = UnlockDataset("X:\\Unlock_dataset\\unlock112",data_transform)
    batch_size = 4
    dataloaders = DataLoader(train_set,batch_size,shuffle=True,num_workers=0)
    test_loaders = DataLoader(test_set,batch_size,shuffle=False,num_workers=0)
    test_loaders_size = len(test_loaders)
    dataloaders_size = len(dataloaders)
    epochs = 200

    ############  选择优化器和损失函数 ########################
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    
    ############  开始训练和评估  #################################
   
    model.eval()
    for idx,(inputs) in enumerate(test_loaders):
        inputs = inputs.to(device)
        out = model(inputs)
        index1 = torch.argmax(out,1)
        correct_sum = torch.sum(index1.cpu() == torch.tensor([1,1,1,1]))
        if correct_sum == batch_size:
            print("end index is {0}".format(idx*batch_size+batch_size//2))
            break


if __name__ == '__main__':
    app.run(debug=True)

