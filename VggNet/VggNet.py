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

def vggnet(fine_tuning,weight_path = None,last_liner_name = "classifier_liner",num_classes = 2,bn = True,model = "D"):
    model = VggNet(last_liner_name,num_classes,bn,model)
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

class VggNet(nn.Module):
    #在__init__初始化函数中写网络架构，这样这个类在被创建时会自动调用__init__函数。从而创建我们需要的网络架构
    def __init__(self,last_liner_name,num_class = 2,bn = True,model = "D"):
        super(VggNet,self).__init__()
        #输入：N*3*224*224
        #print cfg["A"]
        self.features = self.make_layers(cfg[model],bn)

        classifier_order_dict = OrderedDict([("basic_liner1",BasicLiner(7*7*512,4096)),
                                            ("basic_liner2",BasicLiner(4096,4096)),
                                            (last_liner_name,nn.Linear(4096,num_class))])
        self.classifier = nn.Sequential(classifier_order_dict)  #是否戴眼镜二分类

    def forward(self,x):
        #重定义nn.Module类的forward函数，从而实现我们网络的前向传播计算。反向传播不用重写，pytorch会利用autograd机制自动重实现backward函数
        x = self.features(x)
        x = x.view(x.size()[0],-1)  #将feaumap map维度进行转换，转换成batch_size*(w*h*channels) 以适应全连接层的要求
      #  print x.shape
        x = self.classifier(x)
        return x

    def make_layers(self,cfg,bn=True):
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
                basic_conv_3x3 = BasicConv2d3x3(in_channels,v,1,1,bn)
                model_dict["basic_conv_3x3_" + str(i) + "_" + str(j)] = basic_conv_3x3
                j +=1
                in_channels = v

        return nn.Sequential(model_dict)

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
