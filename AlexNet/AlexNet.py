#!/usr/bin/env python
# coding=utf-8

import torch 
import torch.nn as nn
import math
from collections import OrderedDict

def alexnet(fine_tuning,weight_path = None,last_liner_name = "classifier_liner",num_classes = 2,bn = True):
    model = AlexNet(last_liner_name,num_classes,bn)
    if fine_tuning:  #进行fine_tuning
        if weight_path is None: #读取pytorch官方参数
            pass  #暂时不写 这要根据官方的命名来进行调整，比较麻烦
        else:   #读取自己保存的参数
            #print "111"
            checkpoint = torch.load(weight_path)
            pretrain_dict = OrderedDict()
            for key,value in checkpoint["net"].items():
                if key[0:7] == "module.":
                    pretrain_dict[key[7:]] = value
            #print pretrain_dict.keys()
            #for key in pretrain_dict.keys():
            #    print key
            model_dict = model.state_dict()
            #print "222"
            #for key in model_dict.keys():
            #    print key
            pretrain_dict = {k:v for k,v in pretrain_dict.items() if k in model_dict}
            #print pretrain_dict
            model_dict.update(pretrain_dict)
            model.load_state_dict(model_dict)
    return model

class AlexNet(nn.Module):
    #在__init__初始化函数中写网络架构，这样这个类在被创建时会自动调用__init__函数。从而创建我们需要的网络架构
    def __init__(self,last_liner_name,num_classes = 2,bn = True):
        super(AlexNet,self).__init__()
        #输入：N*1*227*227
        self.features = nn.Sequential(
            BasicConv2d11x11(3,96,4,0,bn),  # 输出：N*96*55*55
            nn.MaxPool2d(3,2),  #输出：N*96*27*27
            BasicConv2d5x5(96,256,1,2,bn),  #输出：N*256*27*27
            nn.MaxPool2d(3,2), #输出：N*256*13*13
            BasicConv2d3x3(256,384,1,1,bn), #输出：N*384*13*13
            BasicConv2d3x3(384,384,1,1,bn), #输出：N*384*13*13
            BasicConv2d3x3(384,256,1,1,bn), #输出：N*256*13*13
            nn.MaxPool2d(3,2)  #输出：N*256*6*6
        )

        classifier_order_dict = OrderedDict([("basic_liner1",BasicLiner(6*6*256,4096)), #输出: N*4096
                                            ("basic_liner2",BasicLiner(4096,4096)),#输出：N*4096
                                            (last_liner_name,nn.Linear(4096,num_classes))]) #输出：N*num_classes
        self.classifier = nn.Sequential(classifier_order_dict)
    def forward(self,x):
        #重定义nn.Module类的forward函数，从而实现我们网络的前向传播计算。反向传播不用重写，pytorch会利用autograd机制自动重实现backward函数
        x = self.features(x)
      #  print x.shape
        x = x.view(x.size()[0],-1)
        x = self.classifier(x)
        return x

class BasicConv2d1x1(nn.Module):
    def __init__(self,in_planes,out_planes,stride=1,padding=0,bn=True):
        super(BasicConv2d1x1,self).__init__()
        self.use_bn = bn
        self.conv = nn.Conv2d(in_planes,out_planes,1,stride,padding)
        if use_bn:
            self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU()

    def forward(self,x):
        x = self.conv(x)
        if use_bn:
            x = self.bn(x)
        return self.relu(x)

class BasicConv2d3x3(nn.Module):
    def __init__(self,in_planes,out_planes,stride=1,padding=0,bn=True):
        super(BasicConv2d3x3,self).__init__()
        self.use_bn = bn
        self.conv = nn.Conv2d(in_planes,out_planes,3,stride,padding)
        if self.use_bn:
            self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU()

    def forward(self,x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        return self.relu(x)

class BasicConv2d5x5(nn.Module):
    def __init__(self,in_planes,out_planes,stride=1,padding=0,bn=True):
        super(BasicConv2d5x5,self).__init__()
        self.use_bn = bn
        self.conv = nn.Conv2d(in_planes,out_planes,5,stride,padding)
        if self.use_bn:
            self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU()

    def forward(self,x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        return self.relu(x)


class BasicConv2d7x7(nn.Module):
    def __init__(self,in_planes,out_planes,stride=1,padding=0,bn=True):
        super(BasicConv2d7x7,self).__init__()
        self.use_bn = bn
        self.conv = nn.Conv2d(in_planes,out_planes,7,stride,padding)
        if self.use_bn:
            self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU()

    def forward(self,x):
        x = self.conv(x)
        if bn:
            x = self.bn(x)
        return self.relu(x)

class BasicConv2d11x11(nn.Module):
    def __init__(self,in_planes,out_planes,stride=1,padding=0,bn=True):
        super(BasicConv2d11x11,self).__init__()
        self.use_bn = bn
        self.conv = nn.Conv2d(in_planes,out_planes,11,stride,padding)
        if self.use_bn:
            self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU()

    def forward(self,x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        return self.relu(x)
class BasicLiner(nn.Module):
    def __init__(self,in_planes,out_planes):
        super(BasicLiner,self).__init__()
        self.liner = nn.Linear(in_planes,out_planes)
        self.relu = nn.ReLU()

    def forward(self,x):
        x = self.liner(x)
        return self.relu(x)
