#!/usr/bin/env python
# coding=utf-8

import torch 
import torch.nn as nn

class LeNet(nn.Module):
    #在__init__初始化函数中写网络架构，这样这个类在被创建时会自动调用__init__函数。从而创建我们需要的网络架构
    def __init__(self):
        super(LeNet,self).__init__()
        #输入：N*1*28*28
        self.conv1 = nn.Conv2d(1,20,5)  #参数为：输入数据通道，输出数据通道，卷积核大小(int或tuple),步长(默认为1),padding(默认为0),dilation(效果未知，默认为1),groups(效果未知，默认为1),是否开启偏置项(默认开启) 输出：N*20*24*24
        self.pool1 = nn.MaxPool2d(2,2)  #二维最大池化 参数为：模板大小(int或者tuple),步长(默认为kernel大小),padding(默认为0),dilation(效果未知，默认为1), 输出：N*20*12*12
        self.conv2 = nn.Conv2d(20,50,5)  #输出：N*50*8*8
        self.pool2 = nn.MaxPool2d(2,2) #输出：N*50*4*4
        self.fc1 = nn.Linear(4*4*50,500)  #线性变换层 即全连接层或内积层 输入之前要把每个样本转换成一维向量 参数为：每个样本的输入大小，每个样本的输出大小,是否学习偏置项(默认开启) 输出：N*500
        self.relu = nn.ReLU()  #非线性激活层   输出：N*500
        self.dropout = nn.Dropout(0.5)  #dropout层
        #self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(500,10)  #输出：N*10
    def forward(self,x):
        #重定义nn.Module类的forward函数，从而实现我们网络的前向传播计算。反向传播不用重写，pytorch会利用autograd机制自动重实现backward函数
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = x.view(x.size()[0],-1)  #将feaumap map维度进行转换，转换成batch_size*(w*h*channels) 以适应全连接层的要求 
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        #x = self.sigmoid(x)
        x = self.fc2(x)

        return x


