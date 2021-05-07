import requests

url = 'http://localhost:5000/predict'
epoch = 1000
for i in range(1000):
    r = requests.post(url,json={'facePath': 'Y:\\DeepLearning\\SiameseNetForFace\\data\\2\\0.png'})
    print(r.json())