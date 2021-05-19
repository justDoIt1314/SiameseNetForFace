from os import path
from shutil import Error
import cv2
import torch
from PIL import Image 
import os
import glob
import random
from itertools import combinations,permutations
from facenet_pytorch import MTCNN


class FaceDataset(torch.utils.data.Dataset):
    def __init__(self,root,transforms,classes):
        self.root = root
        self.transforms = transforms
        self.classes = [i for i in range(classes)]
        self.imgs = {}
        combin = combinations(self.classes,2)
        for i in self.classes:
            self.imgs[i] = list(sorted(glob.glob(os.path.join(self.root,str(i))+'/*')))

        
    def __getitem__(self, idx):

        if random.random()>0.5:
            idx1 = random.randint(0,len(self.classes)-1)
            ind1 = random.randint(0,2)
            ind2 = random.randint(0,2)
            img_path_1 = self.imgs[idx1][ind1]
            img_path_2 = self.imgs[idx1][ind2]
            
            img_1 = Image.open(img_path_1).convert("RGB")
            img_2 = Image.open(img_path_2).convert("RGB")

            if self.transforms is not None:
                img_1 = self.transforms(img_1)
                img_2 = self.transforms(img_2)
            return img_1,img_2,torch.ones((1))
        else:
            idx1 = random.randint(0,len(self.classes)-1)
            idx2 = random.randint(0,len(self.classes)-1)
            
            ind1 = random.randint(0,2)
            ind2 = random.randint(0,2)
            img_path_1 = self.imgs[idx1][ind1]
            img_path_2 = self.imgs[idx2][ind2]
            
            img_1 = Image.open(img_path_1).convert("RGB")
            img_2 = Image.open(img_path_2).convert("RGB")

            if self.transforms is not None:
                img_1 = self.transforms(img_1)
                img_2 = self.transforms(img_2)
            return img_1,img_2,torch.zeros((1))
        

    def __len__(self):
        return 1000

class FaceClassDataset(torch.utils.data.Dataset):
    def __init__(self,root,transforms,classes):
        self.root = root
        self.transforms = transforms
        self.classes = classes
        self.imgs = list(sorted(glob.glob(self.root + "/*/*.*")))
        print(self.imgs)

        
    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transforms is not None:
            img = self.transforms(img)
        
        y = int(img_path.split('\\')[-2].split('_')[0])
        return img, torch.tensor(y,dtype=torch.long)


    def __len__(self):
        return len(self.imgs)

class GenderClassDataset(torch.utils.data.Dataset):
    def __init__(self,root,transforms,classes,isTrain):
        self.root = root
        self.transforms = transforms
        self.classes = classes
        self.imgs_Female = list(glob.glob(os.path.join(self.root,'Female') + "/*.*"))
        self.imgs_Male = list(glob.glob(os.path.join(self.root,'Male') + "/*.*"))
        self.imgs = self.imgs_Female+self.imgs_Male
        if not isTrain:
            self.imgs = self.imgs[:len(self.imgs)//4]

        
    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transforms is not None:
            img = self.transforms(img)
        
        y = img_path.split('\\')[-2]
        label = 0 if y == 'Female' else 1

        return img, torch.tensor(label,dtype=torch.long)

def checkAge():
    import numpy
    path_imgs = list(glob.glob("x:\\wiki_crop" + "/*/*"))
    for img_path in path_imgs:
        filename = str(img_path.split('\\')[-1])
        _,birth,now = filename.split('_')
        birth = int(birth.split('-')[0])
        now = int(now.split('.')[0])
        age = now - birth
        if(age<0):
            print("age<0 "+ img_path)
            os.remove(img_path)
        elif(age >100):
            print("age>100 "+ img_path)
            os.remove(img_path)

def ExtractFace():
    import numpy
    path_imgs = list(glob.glob("x:\\wiki_crop" + "/*/*"))
    # for img_path in path_imgs:
    #     filename = str(img_path.split('\\')[-1])
    #     _,birth,now = filename.split('_')
    #     birth = int(birth.split('-')[0])
    #     now = int(now.split('.')[0])
    #     age = now - birth
    #     print(img_path)

    path_imgs = path_imgs[16274:]
    mtcnn = MTCNN(keep_all=True, device=torch.device('cuda:0'))
    for i in range(len(path_imgs)):
        try:
            frame = cv2.imread(path_imgs[i])
            boxes, _ = mtcnn.detect(frame)
            if type(boxes) == numpy.ndarray:
                for bbox in boxes:
                    x1 = int(bbox[0])
                    y1 = int(bbox[1])
                    x2 = int(bbox[2])
                    y2 = int(bbox[3])
                    subImage = frame[y1:y2, x1:x2]
                    cv2.imwrite(path_imgs[i],subImage)
            else:
                if os.path.exists(path_imgs[i]):
                    os.remove(path_imgs[i])
       
        except Exception:
            print(path_imgs[i]+"error")
            if os.path.exists(path_imgs[i]):
                os.remove(path_imgs[i])

class AgeDataset(torch.utils.data.Dataset):
    # X:\wiki_crop
    def __init__(self,root,transforms,classes,isTrain):
        self.root = root
        self.transforms = transforms
        self.classes = classes
        self.imgs = list(glob.glob(self.root + "/*/*"))
        if not isTrain:
            self.imgs = self.imgs[:len(self.imgs)//5]
        else:
            self.imgs = self.imgs[len(self.imgs)//5:]

        
    def __getitem__(self, idx):
        # try:
        img_path = self.imgs[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transforms is not None:
            img = self.transforms(img)
        
        filename = str(img_path.split('\\')[-1])
        _,birth,now = filename.split('_')
        birth = int(birth.split('-')[0])
        now = int(now.split('.')[0])
        age = now - birth
        # print(self.imgs[idx])
        return img, torch.tensor(age,dtype=torch.float32)
        # except Exception:
        #     print("加载图像出错")

    def __len__(self):
        return len(self.imgs)

class UnlockDataset(torch.utils.data.Dataset):
    def __init__(self,root,transforms):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(glob.glob(os.path.join(self.root)+'/*')))
        
    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root,self.imgs[idx])
        img = Image.open(img_path).convert("RGB")

        if self.transforms is not None:
            img = self.transforms(img)

        return img

    def __len__(self):
        return len(self.imgs)

if __name__ == "__main__":
    checkAge()