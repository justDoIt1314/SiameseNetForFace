# encoding=utf-8
import re
import numpy as np
from PIL import Image
 
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.layers import Activation
from keras.layers import Input, Lambda, Dense, Dropout, Convolution2D, MaxPooling2D, Flatten
from keras.models import Sequential, Model
from keras.optimizers import RMSprop
 
 
def read_image(filename, byteorder='>'):
    '''
    读取输入图像，返回一个Numpy数组
    '''
    # 首先将图像以raw格式读入缓冲区
    with open(filename, 'rb') as f:
        buffer = f.read()
 
    # 使用正则表达式获取图片的头部、宽度、高度、最大值
    header, width, height, maxval = re.search(
        b"(^P5\s(?:\s*#.*[\r\n])*"
        b"(\d+)\s(?:\s*#.*[\r\n])*"
        b"(\d+)\s(?:\s*#.*[\r\n])*"
        b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
 
    # 使用np.frombuffer(该函数将缓冲区转换为一维数组)将图像转换为numpy数组
    return np.frombuffer(buffer,
                         dtype='u1' if int(maxval) < 256 else byteorder + 'u2',
                         count=int(width) * int(height),
                         offset=len(header)
                         ).reshape((int(height), int(width)))
 
 
# 举例
Image.open("data/1/1.png")
img = read_image('data/1/1.png')
ind2 = np.random.randint(3)
print(ind2)
 
size = 2
total_sample_size = 100
 
 
def get_data(size, total_sample_size):
    '''
    生成数据，对于孪生网络，数据应该是成对的（正和负），并带有二元标签
    :param size:
    :param total_sample_size:
    :return:
    '''
    # 读取图像
    image = read_image('data/' + str(1) + '/' + str(1) + '.png', 'rw+')
    # 缩减尺寸为原来的一半
    image = image[::size, ::size]
    # 获取新尺寸
    dim1 = image.shape[0]
    dim2 = image.shape[1]
 
    count = 0
 
    # 初始化numpy数组
    x_geuine_pair = np.zeros([total_sample_size, 2, 1, dim1, dim2])  # 2 is for pairs
    y_genuine = np.zeros([total_sample_size, 1])
 
    for i in range(5):
        for j in range(int(total_sample_size / 40)):
            ind1 = 0
            ind2 = 0
 
            # 从同一个目录读取图像（正样本对）
            while ind1 == ind2:
                ind1 = np.random.randint(10)
                ind2 = np.random.randint(10)
 
            # 读取两张图像
            img1 = read_image('data/' + str(i) + '/' + str(ind1) + '.png', 'rw+')
            img2 = read_image('data/' + str(i) + '/' + str(ind2) + '.png', 'rw+')
 
            # 缩减尺寸
            img1 = img1[::size, ::size]
            img2 = img2[::size, ::size]
 
            # 将图像存入初始化好的numpy数组中
            x_geuine_pair[count, 0, 0, :, :] = img1
            x_geuine_pair[count, 1, 0, :, :] = img2
 
            # 因为从同一目录中取出，所以为正样本对，标签为1
            y_genuine[count] = 1
            count += 1
 
    count = 0
    x_imposite_pair = np.zeros([total_sample_size, 2, 1, dim1, dim2])
    y_imposite = np.zeros([total_sample_size, 1])
 
    for i in range(int(total_sample_size / 10)):
        for j in range(10):
 
            # 从不同的目录读取图像（负样本对）
            while True:
                ind1 = np.random.randint(40)  # 40以内的随机数
                ind2 = np.random.randint(40)
                if ind1 != ind2:
                    break
 
            img1 = read_image('data/orl_faces/s' + str(ind1 + 1) + '/' + str(j + 1) + '.pgm', 'rw+')
            img2 = read_image('data/orl_faces/s' + str(ind2 + 1) + '/' + str(j + 1) + '.pgm', 'rw+')
 
            img1 = img1[::size, ::size]
            img2 = img2[::size, ::size]
 
            x_imposite_pair[count, 0, 0, :, :] = img1
            x_imposite_pair[count, 1, 0, :, :] = img2
            # 因为是从不同的目录取出，所以分配标签为0
            y_imposite[count] = 0
            count += 1
 
    # 将正样本对和负样本对拼接
    X = np.concatenate([x_geuine_pair, x_imposite_pair], axis=0) / 255
    Y = np.concatenate([y_genuine, y_imposite], axis=0)
 
    return X, Y
 
 
X, Y = get_data(size, total_sample_size)
# 生成完数据并检查数据大小，拼接完后，有10000个正样本对和10000个负样本对
 
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=.25)
#将数据打乱分成75%的训练数据和25%的测试数据
 
def build_base_network(input_shape):
    '''
    构建孪生网络，首先定义基网络，是一个特征提取的卷积网络，建立两个带有Relu激活函数和最大池化的卷积层和扁平化层
    '''
    seq = Sequential()
 
    nb_filter = [6, 12]
    kernel_size = 3
 
    # convolutional layer 1
    seq.add(Convolution2D(nb_filter[0], kernel_size, kernel_size, input_shape=input_shape,
                          border_mode='valid', dim_ordering='th'))
    seq.add(Activation('relu'))
    seq.add(MaxPooling2D(pool_size=(2, 2)))
    seq.add(Dropout(.25))
 
    # convolutional layer 2
    seq.add(Convolution2D(nb_filter[1], kernel_size, kernel_size, border_mode='valid', dim_ordering='th'))
    seq.add(Activation('relu'))
    seq.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='th'))
    seq.add(Dropout(.25))
 
    # flatten
    seq.add(Flatten())
    seq.add(Dense(128, activation='relu'))
    seq.add(Dropout(0.1))
    seq.add(Dense(50, activation='relu'))
    return seq
 
 
input_dim = x_train.shape[2:]
img_a = Input(shape=input_dim)
img_b = Input(shape=input_dim)
 
base_network = build_base_network(input_dim)
feat_vecs_a = base_network(img_a)
feat_vecs_b = base_network(img_b)
# 得到图像对的特征向量，接下来输入欧氏距离能量函数，计算出距离
 
def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))
 
 
def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)
 
 
distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([feat_vecs_a, feat_vecs_b])
 
epochs = 13
rms = RMSprop()
model = Model(input=[img_a, img_b], output=distance)
 
 
def contrastive_loss(y_true, y_pred):
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))
 
 
model.compile(loss=contrastive_loss, optimizer=rms)
 
img_1 = x_train[:, 0]
img2 = x_train[:, 1]
 
model.fit([img_1, img2], y_train, validation_split=.25,
          batch_size=128, verbose=2, nb_epoch=epochs)
 
pred = model.predict([x_test[:, 0], x_test[:, 1]])
 
 
def compute_accuracy(predictions, labels):
    return labels[predictions.ravel() < 0.5].mean()
 
compute_accuracy(pred, y_test)