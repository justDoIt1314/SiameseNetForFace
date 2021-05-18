from flask import jsonify
import torch.nn.functional as F
from torchvision import datasets,models,transforms
from facenet_pytorch import MTCNN
import torch
import numpy
import numpy as np
import cv2
from PIL import Image, ImageDraw
import time
import requests
import os
import cv2
import time
import base64
from network import LoadInceptionNet,LoadInceptionNetForAge
import glob
# Determine if an nvidia GPU is available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda:0')
print('Running on device: {}'.format(device))
url = 'http://localhost:5000/test_gender'
gender = ['Female','Male']
device = torch.device('cuda:0')

data_transform = transforms.Compose([
    transforms.Resize((299,299)),
    transforms.RandomRotation(20),
    transforms.ToTensor()
])

def imgToBytes(frame):
    # 编码图像为 bytes
    success, encoded_image = cv2.imencode(".jpg", frame)
    img_bytes = encoded_image.tobytes()
    img_bytes = base64.b64encode(img_bytes)
    img_bytes = img_bytes.decode('ascii')
    return img_bytes


def get_gender_model():
    model = LoadInceptionNet(2)
    model.load_state_dict(torch.load('./gender.pth'))
    return model
gender_model = get_gender_model()
gender_model.to(device)

age_model = LoadInceptionNetForAge(1)
age_model.load_state_dict(torch.load('age_predict.pth'))
age_model.to(device)

def detectGender(image_data):
    if np.shape(image_data)[0] == 0 or np.shape(image_data)[1] == 0 or np.shape(image_data)[2] == 0:
        return 'unknow'
    image = Image.fromarray(cv2.cvtColor(image_data,cv2.COLOR_BGR2RGB))
    
    image = data_transform(image)
    inputs = torch.unsqueeze(image,0)
    inputs = inputs.to(device)
    gender_model.eval()
    out = gender_model(inputs)
    out = out[0]
    print(out)
    res = F.softmax(out)
    index = torch.argmax(res,0)
    if res[index.item()] < 0.8:
        res = 'unknow'
    else:
        res = gender[index.item()]

    age_model.eval()
    age = age_model(inputs)
    
    return res,int(age.item())

def main2():
    imgs = list(sorted(glob.glob('Y:\\DeepLearning\\SiameseNetForFace\\training_images_bigdata'+ "\\Female\\*.jpg")))
    for filepath in imgs:
        #frame_draw = cv2.imread(filepath).copy()
        frame_draw = Image.open(filepath).convert('RGB')
        res = detectGender(frame_draw)
        #cv2.putText(frame_draw, str(res), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)
        #cv2.imshow('face',frame_draw)
        #print(res)
        cv2.waitKey()
def main(Isvideo):
    mtcnn = MTCNN(keep_all=True, device=device)
    cap = None
    if Isvideo:
        cap = cv2.VideoCapture('X:\\mda-mb5g90fthhs8pttt.mp4')
    else:
        cap = cv2.VideoCapture(0)
    frames_tracked = []
    count = 0
    cv2.namedWindow('face',cv2.WINDOW_NORMAL)
    while True:
        ret, frame = cap.read()
        frame = cv2.imread("20210518152653.jpg")
        if not ret:
            return
        count += 1
        if count % 5 == 0:
            # Detect faces
            boxes, _ = mtcnn.detect(frame)
            # Draw faces
            frame_draw = frame.copy()
            if type(boxes) == numpy.ndarray:
                for bbox in boxes:
                    x1 = int(bbox[0])
                    y1 = int(bbox[1])
                    p1 = (x1,y1)
                    x2 = int(bbox[2])
                    y2 = int(bbox[3])
                    p2 = (x2, y2)
                    subImage = frame[y1:y2, x1:x2]
                    # img_bytes = imgToBytes(subImage)
                    # 上传图像 等待回传
                    # data = {'img': img_bytes}
                    # r = requests.post(url, data=data)
                    try:
                        r ,age= detectGender(subImage)
                    except Exception:
                        print("error")
                        continue
                    #r = r.json()
                    cv2.rectangle(frame_draw, p1, p2, (255,0,0), 2, 1)
                    cv2.putText(frame_draw, "gender:"+str(r)+" age:"+str(age), (int(bbox[0]-40),int(bbox[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 2)
            cv2.imshow('face',frame_draw)
            k = cv2.waitKey(1) & 0xff
            if k == 27 : break
            print('ok')


    

if __name__ == '__main__':
    main(False)




# def main():
#     mtcnn = MTCNN(keep_all=True, device=device)
    
#     cap = cv2.VideoCapture(0)
#     frames_tracked = []
#     while True:
#         ret, frame = cap.read()
        
#         if not ret:
#             return
#         # Detect faces
#         boxes, _ = mtcnn.detect(frame)
#         # Draw faces
#         frame_draw = frame.copy()
#         if type(boxes) == numpy.ndarray:
#             for bbox in boxes:
#                 p1 = (int(bbox[0]), int(bbox[1]))
#                 # p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
#                 p2 = (int(bbox[2]), int(bbox[3]))
                
#                 cv2.rectangle(frame_draw, p1, p2, (255,0,0), 2, 1)
#                 #frame_draw.rectangle(box.tolist(), outline=(255, 0, 0), width=6)
#         cv2.imshow('face',frame_draw)
#         k = cv2.waitKey(1) & 0xff
#         if k == 27 : break
#         print('ok')

# if __name__ == '__main__':
#     main()