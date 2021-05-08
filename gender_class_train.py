import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as v_utils
import matplotlib.pyplot as plt
import cv2
import math
import copy
import time
import random
import torchvision
from torchvision.models import resnet50,alexnet,vgg16,inception_v3
from torch.utils.data import TensorDataset,DataLoader
from torchvision import datasets,models,transforms
from dataset import FaceDataset,UnlockDataset,FaceClassDataset,GenderClassDataset
from network import myModle,myModle_class,LoadCNN,InceptionNet,LoadInceptionNet
from sklearn.model_selection import train_test_split
nn.L1Loss
batch = 4
faceClass = 2
epochs = 1000
data_transform = transforms.Compose([
        transforms.Resize((299,299)),
        transforms.RandomRotation(20),
        transforms.ToTensor()
    ])



def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = "gender.pth"
    train_set = GenderClassDataset("Y:\\DeepLearning\\SiameseNetForFace\\chinese_face_data",data_transform,faceClass,True)
    test_set = GenderClassDataset("Y:\\DeepLearning\\SiameseNetForFace\\chinese_face_data",data_transform,faceClass,False)

    test_dataloader = DataLoader(test_set,batch,shuffle=False,num_workers=4)
    
    
    dataloader = DataLoader(train_set,batch,shuffle=True,num_workers=4)
    dataloader_size = len(dataloader)


    #model = myModle_class(faceClass)
    
    model = LoadInceptionNet(faceClass)
    
    if os.path.exists(model_path):
        #model = torch.load(model_path)
        model.load_state_dict(torch.load(model_path))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):

        model.eval()
        print("test>>>")
        error_nums = 0
        for idx,(imgs_1,labels) in enumerate(test_dataloader):
            imgs_1 = imgs_1.to(device)
            labels = labels.to(device)
            outs_1 = model(imgs_1)
            res = torch.argmax(outs_1,1)
            error_nums += torch.sum(torch.abs(torch.sub(res,labels)))
            print("epoch: {0}/{1}, batch:{2}/{3},error_num:{4}".format(epoch,epochs,idx,len(test_dataloader),error_nums))
        
        err = error_nums.item() / (batch*len(test_dataloader))
        print("accuracy: "+ str(1-err))
        model.train()
        print("train>>>")
        for idx,(imgs_1,labels) in enumerate(dataloader):
            imgs_1 = imgs_1.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outs_1,hidd = model(imgs_1)
            loss = criterion(outs_1,labels)
            loss.backward()
            optimizer.step()
            print("epoch: {0}/{1}, batch:{2}/{3} loss: {4}".format(epoch,epochs,idx,dataloader_size,loss.item()))

        

        
        torch.save(model.state_dict(),model_path)
if __name__ == "__main__":
    main()


    