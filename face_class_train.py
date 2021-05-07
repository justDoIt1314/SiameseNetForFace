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
from torchvision.models import resnet50,alexnet,vgg16
from torch.utils.data import TensorDataset,DataLoader
from torchvision import datasets,models,transforms
from dataset import FaceDataset,UnlockDataset,FaceClassDataset
from network import myModle,myModle_class,LoadCNN
nn.L1Loss
batch = 8
faceClass = 7
epochs = 1000
data_transform = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.RandomRotation(20),
        transforms.ToTensor()
    ])



def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = "class_face.pth"
    train_set = FaceClassDataset("Y:\\DeepLearning\\SiameseNetForFace\\data",data_transform,faceClass)
    dataloader = DataLoader(train_set,batch,shuffle=False,num_workers=4)
    dataloader_size = len(dataloader)


    #model = myModle_class(faceClass)
    model = LoadCNN(faceClass)
    if os.path.exists(model_path):
        #model = torch.load(model_path)
        model.load_state_dict(torch.load(model_path))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        for idx,(imgs_1,labels) in enumerate(dataloader):
                imgs_1 = imgs_1.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outs_1 = model(imgs_1)
                loss = criterion(outs_1,labels)
                loss.backward()
                optimizer.step()
                print("epoch: {0}/{1}, batch:{2}/{3} loss: {4}".format(epoch,epochs,idx,dataloader_size,loss.item()))

        if (epoch+1) % 50 == 0:
            torch.save(model.state_dict(),model_path)
if __name__ == "__main__":
    main()


    