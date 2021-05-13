import torch
from PIL import Image 
import os
import glob
import random
from itertools import combinations,permutations
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