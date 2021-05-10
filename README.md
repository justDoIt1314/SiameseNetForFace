# SiameseNetForFace
用于人脸分类，性别分类，人脸相似度验证。

app.py: 使用Flask 框架，作为服务端为客服端提供识别服务

face_class_train.py:人脸分类训练代码

gender_class_train.py:性别分类的训练代码

facenet.py: 利用MTCNN进行人脸检测，并通过检测的人脸进行性别分类

gender.pth:性别分类的模型文件

network.py:网络构建的相关代码

dataset.py:数据加载的相关代码

siameseTrain.py:使用孪生网络，利用对比损失进行相似度训练

request.py: 客户端代码
