from torchvision.models import resnet50,resnet18,AlexNet
from torch import nn
import torch
from torchvision import datasets,models,transforms
import math
import torch.utils.model_zoo as model_zoo

class myModle(nn.Module):
    def __init__(self,num_classes):
        super(myModle,self).__init__()
        self.dense = nn.Sequential(nn.Linear(1000,num_classes))
        self.resnet50 = resnet50()
        self.resnet18 = resnet18()
        self.alxnet = AlexNet() 
        

    def forward(self,x):
        x = self.resnet50(x)
        x = self.dense(x)
        return x


#Bottleneck是一个class 里面定义了使用1*1的卷积核进行降维跟升维的一个残差块，可以在github resnet pytorch上查看
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

def LoadCNN(faceClass):
    # 加载model
    resnet50 = models.resnet50()
    resnet50.load_state_dict(torch.load("X:\\Downloads\\resnet50-19c8e357.pth"))
    #3 4 6 3 分别表示layer1 2 3 4 中Bottleneck模块的数量。res101则为3 4 23 3 
    cnn = CNN(Bottleneck, [3, 4, 6, 3],faceClass)
    # 读取参数
    pretrained_dict = resnet50.state_dict()
    model_dict = cnn.state_dict()
    # 将pretrained_dict里不属于model_dict的键剔除掉
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 更新现有的model_dict
    model_dict.update(pretrained_dict)
    # 加载我们真正需要的state_dict
    cnn.load_state_dict(model_dict)
    return cnn

#不做修改的层不能乱取名字，否则预训练的权重参数无法传入
class CNN(nn.Module):

    def __init__(self, block, layers, num_classes=9):
        self.inplanes = 64
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        # 新增一个反卷积层
        self.convtranspose1 = nn.ConvTranspose2d(2048, 2048, kernel_size=3, stride=1, padding=1, output_padding=0,
                                                 groups=1, bias=False, dilation=1)
        # 新增一个最大池化层
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        # 去掉原来的fc层，新增一个fclass层
        self.fclass = nn.Linear(2048, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
	#这一步用以设置前向传播的顺序，可以自行调整，前提是合理
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        # 新加层的forward
        # x = x.view(x.size(0), -1)
        # x = self.convtranspose1(x)
        # x = self.maxpool2(x)
        # x = x.view(x.size(0), -1)

        x = torch.flatten(x, 1)
        x = self.fclass(x)

        return x




class myModle_class(nn.Module):
    def __init__(self,num_classes):
        super(myModle_class,self).__init__()
        self.dense = nn.Sequential(nn.Linear(1000,num_classes))
        self.resnet50 = resnet50()
        self.resnet18 = resnet18()
        self.alxnet = AlexNet()
         
        
        

    def forward(self,x):
        x = self.resnet50(x)
        x = self.dense(x)
        return x


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function.
    Based on:
    """

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def check_type_forward(self, in_types):
        assert len(in_types) == 3

        x0_type, x1_type, y_type = in_types
        assert x0_type.size() == x1_type.shape
        assert x1_type.size()[0] == y_type.shape[0]
        assert x1_type.size()[0] > 0
        assert x0_type.dim() == 2
        assert x1_type.dim() == 2
        assert y_type.dim() == 1

    def forward(self, x0, x1, y):
        self.check_type_forward((x0, x1, y))

        # euclidian distance
        diff = x0 - x1
        dist_sq = torch.sum(torch.pow(diff, 2), 1)
        dist = torch.sqrt(dist_sq)

        mdist = self.margin - dist
        dist = torch.clamp(mdist, min=0.0)
        loss = y * dist_sq + (1 - y) * torch.pow(dist, 2)
        loss = torch.sum(loss) / 2.0 / x0.size()[0]
        return loss