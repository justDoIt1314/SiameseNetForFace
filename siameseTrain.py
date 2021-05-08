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
from torchvision.models import resnet50,alexnet
from torch.utils.data import TensorDataset,DataLoader
from torchvision import datasets,models,transforms
from dataset import FaceDataset,UnlockDataset
from network import myModle,ContrastiveLoss

nn.L1Loss
batch = 8
faceClass = 5
epochs = 1000
data_transform = transforms.Compose([
        transforms.Resize((128,128)),
        # transforms.RandomRotation(20),
        # transforms.RandomVerticalFlip(),
        transforms.ToTensor()
    ])
def contrastive_loss(y,d):
    margin = 1
    sub = margin - d
    zeros = torch.zeros((len(y),1)).to(torch.device('cuda'))
    max_value = torch.max(torch.cat((sub,zeros),1),1)[0]
    max_value = torch.unsqueeze(max_value,1)
    loss = y*(d**2)+(1-y)*(max_value**2)
    loss = torch.mean(loss)
    return loss


def distance_euclidean(input1,input2):
    return F.pairwise_distance(input1,input2)
    # inpu1,input2 = vects
    # diff = input1-input2
    # dist_sq = torch.sum(torch.pow(diff, 2), 1)
    # dist = torch.sqrt(dist_sq)
    # return dist
    # return F.cosine_similarity(input1,input2)

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = "face.pth"
    train_set = FaceDataset("Y:\\DeepLearning\\SiameseNetForFace\\data",data_transform,faceClass)
    dataloader = DataLoader(train_set,batch,shuffle=False,num_workers=4)
    dataloader_size = len(dataloader)


    model = myModle(512)
    if os.path.exists(model_path):
        model = torch.load(model_path)
    model.to(device)
    criterion = ContrastiveLoss()
    optimizer = torch.optim.Adam(model.parameters(),0.0001)
    
    for epoch in range(epochs):
        model.train()
        for idx,(imgs_1,imgs_2,labels) in enumerate(dataloader):
                imgs_1 = imgs_1.to(device)
                imgs_2 = imgs_2.to(device)
                #labels = torch.squeeze(labels,1)
                labels = labels.to(device)
                
                outs_1 = model(imgs_1)
                outs_2 = model(imgs_2)
                # 测试
                distance = distance_euclidean(outs_1,outs_2)
                distance = torch.unsqueeze(distance,1)
                # loss = criterion(outs_1,outs_2,labels)
                loss = contrastive_loss(labels,distance)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print("epoch: {0}/{1}, batch:{2}/{3} loss: {4}".format(epoch,epochs,idx,dataloader_size,loss.item()))

        if (epoch+1) % 20 == 0:
            torch.save(model,model_path)
if __name__ == "__main__":
    main()


    