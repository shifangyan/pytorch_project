#!/usr/bin/env python
# coding=utf-8

import torch 
import torch.nn as nn
import math

class AlexNet(nn.Module):
    #在__init__初始化函数中写网络架构，这样这个类在被创建时会自动调用__init__函数。从而创建我们需要的网络架构
    def __init__(self):
        super(AlexNet,self).__init__()
        #输入：N*1*28*28
        self.conv1 = nn.Conv2d(3,96,11,4)  #参数为：输入数据通道，输出数据通道，卷积核大小(int或tuple),步长(默认为1),padding(默认为0),dilation(效果未知，默认为1),groups(效果未知，默认为1),是否开启偏置项(默认开启) 输出：N*96*55*55
        self.bn1 = nn.BatchNorm2d(96)
        self.relu1 = nn.ReLU() 
        self.pool1 = nn.MaxPool2d(3,2)  #二维最大池化 参数为：模板大小(int或者tuple),步长(默认为kernel大小),padding(默认为0),dilation(效果未知，默认为1), 输出：N*96*55*55
        self.conv2 = nn.Conv2d(96,256,5,1,2)  #输出：N*256*27*27
        self.bn2 = nn.BatchNorm2d(256)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(3,2) #输出：N*256*13*13
        self.conv3 = nn.Conv2d(256,384,3,1,1) #输出：N*384*13*13
        self.bn3 = nn.BatchNorm2d(384)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(384,384,3,1,1) #输出：N*384*13*13
        self.bn4 = nn.BatchNorm2d(384)
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(384,256,3,1,1) #输出：N*256*13*13
        self.bn5 = nn.BatchNorm2d(256)
        self.relu5 = nn.ReLU()
        self.pool5 = nn.MaxPool2d(3,2)  #输出：N*256*6*6
        self.fc6 = nn.Linear(6*6*256,4096)  #线性变换层 即全连接层或内积层 输入之前要把每个样本转换成一维向量 参数为：每个样本的输入大小，每个样本的输出大小,是否学习偏置项(默认开启) 输出：N*4096
        self.relu6 = nn.ReLU()  #非线性激活层   输出：N*4096
     #   self.dropout6 = nn.Dropout(0.5)  #dropout层
        self.fc7 = nn.Linear(4096,4096)  #输出：N*4096
        self.relu7 = nn.ReLU()
     #   self.dropout7 = nn.Dropout(0.5)
        self.fc8_glassess = nn.Linear(4096,2)  #是否戴眼镜分类  二分类
    def forward(self,x):
        #重定义nn.Module类的forward函数，从而实现我们网络的前向传播计算。反向传播不用重写，pytorch会利用autograd机制自动重实现backward函数
        x = self.conv1(x)
        x = self.bn1(x)
      #  print x.shape
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
      #  print x.shape
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
      #  print x.shape
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.conv5(x)
        x= self.bn5(x)
        x = self.relu5(x)
        x = self.pool5(x)
      #  print x.shape
        x = x.view(x.size()[0],-1)  #将feaumap map维度进行转换，转换成batch_size*(w*h*channels) 以适应全连接层的要求
      #  print x.shape
        x = self.fc6(x)
        x = self.relu6(x)
      #  x = self.dropout6(x)
        x = self.fc7(x)
        x = self.relu7(x)
      #  x = self.dropout7(x)
        #x = self.sigmoid(x)
        x = self.fc8_glassess(x)

        return x

    def initialize_parameters(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0,math.sqrt(2./n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m,nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m,nn.Linear):
                m.weight.data.normal_(0,0.01)
                m.bias.data.zero_()
