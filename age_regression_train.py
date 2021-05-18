import numpy as np
import os
import sys
from tensorboardX import SummaryWriter
writer = SummaryWriter()
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
from dataset import FaceDataset,UnlockDataset,FaceClassDataset,GenderClassDataset,AgeDataset
from network import myModle,myModle_class,LoadCNN,InceptionNet,LoadInceptionNet,LoadInceptionNetForAge
from sklearn.model_selection import train_test_split
nn.L1Loss
batch = 8
faceClass = 1
epochs = 1000
data_transform = transforms.Compose([
        transforms.Resize((299,299)),
        transforms.RandomRotation(20),
        transforms.ToTensor()
    ])



def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_path = "age_predict.pth"
    train_set = AgeDataset("X:\\wiki_crop",data_transform,faceClass,True)
    test_set = AgeDataset("X:\\wiki_crop",data_transform,faceClass,False)

    test_dataloader = DataLoader(test_set,batch,shuffle=False,num_workers=4)
    len_test_dataloader = len(test_dataloader)
    
    dataloader = DataLoader(train_set,batch,shuffle=True,num_workers=4)
    dataloader_size = len(dataloader)

    model = LoadInceptionNetForAge(faceClass)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.SmoothL1Loss()
    for epoch in range(epochs):

        
        
        model.train()
        print("train>>>")
        for idx,(imgs_1,labels) in enumerate(dataloader):
            imgs_1 = imgs_1.to(device)
            labels = labels.to(device)
            labels = labels.unsqueeze(1)
            optimizer.zero_grad()
            outs_1,hidd = model(imgs_1)
            train_loss = criterion(outs_1,labels)
            train_loss.backward()
            optimizer.step()
            print("epoch: {0}/{1}, batch:{2}/{3} train_loss: {4}".format(epoch,epochs,idx,dataloader_size,train_loss.item()))
            writer.add_scalar("train_loss", train_loss.item(),idx + epoch * dataloader_size)

        
        model.eval()
        print("test>>>")
        for idx,(imgs_1,labels) in enumerate(test_dataloader):
            imgs_1 = imgs_1.to(device)
            labels = labels.to(device)
            labels = labels.unsqueeze(1)
            outs_1 = model(imgs_1)
            test_loss = F.smooth_l1_loss(outs_1,labels)
            print("epoch: {0}/{1}, batch:{2}/{3},test_loss:{4}".format(epoch,epochs,idx,len(test_dataloader),test_loss))
            writer.add_scalar("test_loss", test_loss.item(),idx + epoch * len_test_dataloader)
        
        torch.save(model.state_dict(),model_path)

if __name__ == "__main__":
    main()


    