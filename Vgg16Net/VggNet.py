#!/usr/bin/env python
# coding=utf-8

import torch 
import torch.nn as nn
from collections import OrderedDict

cfg = {"A":[64,"M",128,"M",256,256,"M",512,512,"M",512,512,"M"], #VGG11
       "B":[64,64,"M",128,128,"M",256,256,"M",512,512,"M",512,512,"M"], #VGG13
       "D":[64,64,"M",128,128,"M",256,256,256,"M",512,512,512,"M",512,512,512,"M"], #VGG16
       "E":[64,64,"M",128,128,"M",256,256,256,256,"M",512,512,512,512,"M",512,512,512,512,"M"], #VGG19
      }

class VggNet(nn.Module):
    #在__init__初始化函数中写网络架构，这样这个类在被创建时会自动调用__init__函数。从而创建我们需要的网络架构
    def __init__(self,model= "D",num_class = 2,batch_norm = False):
        super(VggNet,self).__init__()
        #输入：N*3*224*224
        #print cfg["A"]
        self.features = self.make_layers(cfg[model],batch_norm)
    #    self.features = nn.Sequential(nn.Conv2d(3,64,3,1,1), #参数为：输入数据通道，输出数据通道，卷积核大小(int或tuple),步长(默认为1),padding(默认为0),dilation(效果未知，默认为1),groups(效果未知，默认为1),是否开启偏置项(默认开启) 输出：N*64*224*224
    #                                  nn.BatchNorm2d(64),
    #                                  nn.ReLU(),
    #                                #  nn.Conv2d(64,64,3,1,1), #输出：N*64*224*224
    #                                #  nn.ReLU(),
    #                                  nn.MaxPool2d(2,2), #输出：N*64*112*112
    #                                  nn.Conv2d(64,128,3,1,1), #输出：N*128*112*112
    #                                  nn.BatchNorm2d(128),
    #                                  nn.ReLU(),
    #                                #  nn.Conv2d(128,128,3,1,1), #输出：N*128*112*112
    #                                #  nn.ReLU(),
    #                                  nn.MaxPool2d(2,2), #输出：N*128*56*56
    #                                  nn.Conv2d(128,256,3,1,1), #输出:N*256*56*56
    #                                  nn.BatchNorm2d(256),
    #                                  nn.ReLU(),
    #                                  nn.Conv2d(256,256,3,1,1),
    #                                  nn.BatchNorm2d(256),
    #                                  nn.ReLU(),
    #                                #  nn.Conv2d(256,256,3,1,1),
    #                                #  nn.ReLU(),
    #                                  nn.MaxPool2d(2,2), #输出：N*256*28*28
    #                                  nn.Conv2d(256,512,3,1,1), #输出：N*512*28*28
    #                                  nn.BatchNorm2d(512),
    #                                  nn.ReLU(),
    #                                  nn.Conv2d(512,512,3,1,1),
    #                                  nn.BatchNorm2d(512),
    #                                  nn.ReLU(),
    #                                #  nn.Conv2d(512,512,3,1,1),
    #                                #  nn.ReLU(),
    #                                  nn.MaxPool2d(2,2), #输出：N*512*14*14
    #                                  nn.Conv2d(512,512,3,1,1), #输出：N*512*14*14
    #                                  nn.BatchNorm2d(512),
    #                                  nn.ReLU(),
    #                                  nn.Conv2d(512,512,3,1,1),
    #                                  nn.BatchNorm2d(512),
    #                                  nn.ReLU(),
    #                                #  nn.Conv2d(512,512,3,1,1), 
    #                                #  nn.ReLU(),
    #                                  nn.MaxPool2d(2,2)) #输出：N*512*7*7


        self.classifier = nn.Sequential(OrderedDict([("fc1",nn.Linear(7*7*512,4096)),  #线性变换层 即全连接层或内积层 输入之前要把每个样本转换成一维向量 参数为：每个样本的输入大小，每个样本的输出大小,是否学习偏置项(默认开启) 输出：N*4096
                                      ("fc_relu1",nn.ReLU()),
                                      ("fc2",nn.Linear(4096,4096)), #输出：N*4096
                                      ("fc_relu2",nn.ReLU()),
                                      ("fc3",nn.Linear(4096,num_class))])) #是否戴眼镜二分类

    def forward(self,x):
        #重定义nn.Module类的forward函数，从而实现我们网络的前向传播计算。反向传播不用重写，pytorch会利用autograd机制自动重实现backward函数
        x = self.features(x)
        x = x.view(x.size()[0],-1)  #将feaumap map维度进行转换，转换成batch_size*(w*h*channels) 以适应全连接层的要求
      #  print x.shape
        x = self.classifier(x)
        return x

    def make_layers(self,cfg,batch_norm=False):
        model_dict = OrderedDict()
        in_channels = 3
        i = 1
        j = 1
       # print cfg
        for v in cfg:
            if v == "M":
                maxpool = nn.MaxPool2d(2,2)
                model_dict["maxpool"+str(i)] = maxpool
                i += 1
                j = 1
            else:
                conv2d = nn.Conv2d(in_channels,v,3,1,1)
                model_dict["conv-"+str(i)+"_"+str(j)] = conv2d
                if batch_norm:
                    bn = nn.BatchNorm2d(v)
                    model_dict["bn-"+str(i)+"_"+str(j)] = bn
                relu = nn.ReLU()
                model_dict["relu-"+str(i)+"_"+str(j)] = relu
                j +=1
                in_channels = v

        return nn.Sequential(model_dict)

