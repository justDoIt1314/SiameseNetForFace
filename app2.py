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
from network import myModle_class,LoadCNN,LoadInceptionNet
from PIL import Image 


app = Flask(__name__)
#app.config['JSONIFY_MIMETYPE'] ="application/json;charset=utf-8"   # 指定浏览器渲染的文件类型，和解码格式；
device = torch.device('cuda:0' if not torch.cuda.is_available() else 'cpu')
device = torch.device('cuda:0')

faceClasses = 7
#names = ['唐僧','姚明','张学友','曹寅','郭爱斌','饶志豪', '曹焕琪']
names = ['tangseng','yaoming','zhangxueyou','caoyin','guoaibin','raozhihao','caohuanqi']
gender = ['Female','Male']
def get_class_face_model():
    #model = myModle_class(faceClasses)
    model = LoadCNN(faceClasses)
    #model = torch.load('./class_face.pth')
    model.load_state_dict(torch.load('./class_face.pth'))
    return model
def get_gender_model():
    model = LoadInceptionNet(2)
    model.load_state_dict(torch.load('./gender.pth'))
    return model

model = get_gender_model()
model.to(device)


data_transform = transforms.Compose([
    transforms.Resize((299,299)),
    transforms.RandomRotation(20),
    transforms.ToTensor()
])

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict2', methods=['POST'])
def predict2():
    data = request.get_json(force=True)
    base64_decoded = base64.b64decode(data['imageData'])
    
    img = Image.open(io.BytesIO(base64_decoded)).convert("RGB")
    inputs = data_transform(img)
    inputs = torch.unsqueeze(inputs,0)

    model.eval()
    inputs = inputs.to(device)
    out, hidd = model(inputs)
    res = F.softmax(out[0])
    index = torch.argmax(res,0)
    if res[index.item()] < 0.5:
        name = 'error'
    else:
        name = names[index.item()]
    
    #return name
    return jsonify({'result': name})
@app.route('/test_gender', methods=['POST'])
def test_gender():
    img = request.form.get('img')

    if img:
        # 解码图像数据
        img = base64.b64decode(img.encode('ascii'))
        image_data = np.fromstring(img, np.uint8)
        image_data = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        image = Image.fromarray(cv2.cvtColor(image_data,cv2.COLOR_BGR2RGB))
        image = data_transform(image)
        inputs = torch.unsqueeze(image,0)
        inputs = inputs.to(device)
        model.eval()
        out = model(inputs)
        out = out[0]
        res = F.softmax(out)
        index = torch.argmax(res,0)
        if res[index.item()] < 0.5:
            res = 'unknow'
        else:
            res = gender[index.item()]
        return jsonify({'result': res})
    else:
        return jsonify({'fail': res})




if __name__ == '__main__':
    app.run(debug=True)

