#!/usr/bin/env python
# coding=utf-8
from InceptionV3 import *
import torch
import torch.nn as nn
import torch.nn.functional as F

class HydraPlusNet(nn.Module):
    def __init__(self,num_class,is_fusion=True):
        super(HydraPlusNet,self).__init__()
        self.is_fusion = is_fusion
        self.MNet = Inception3(num_class,is_fusion,False) #MNet
        if is_fusion:
            self.AF1 = AF1()
            self.AF2 = AF2()
            self.AF3 = AF3()

            self.fc_ = nn.Linear(2048*73,num_class)

    def forward(self,x):
        if self.is_fusion:
            x,y,z = self.MNet(x)
            F1 = self.AF1([x,y,z])
            F2 = self.AF2([x,y,z])
            F3 = self.AF3([x,y,z])

            ret = torch.cat((z,F1,F2,F3),dim=1)

            ret = F.avg_pool2d(ret,kernel_size=8)
            ret = F.dropout(ret,training = self.training)
            ret = ret.view(ret.size(0),-1)
            ret = self.fc_(ret)

        else:
            ret = self.MNet(x)
        
        return ret
class AF1(nn.Module):
    def __init__(self):
        super(AF1,self).__init__()
        self.att = BasicConv2d(288,8,kernel_size =1) 
        self.incep2 = nn.Sequential(InceptionB(288),InceptionC(768,channels_7x7 = 128),InceptionC(768,channels_7x7=160),InceptionC(768,channels_7x7=160),InceptionC(768,channels_7x7=192))
        self.incep3 = nn.Sequential(InceptionD(768),InceptionE(1280),InceptionE(2048))
        self.incep3_2 = nn.Sequential(InceptionD(768),InceptionE(1280),InceptionE(2048))
    def forward(self,input):
        x,y,z = input
        attentive = self.att(x)
        ret = 0
        for i in range(8):
            temp = attentive[:,i].clone()
            temp = temp.view(-1,1,35,35).expand(-1,288,35,35)
            R1 = x *temp
            R1 = self.incep2(R1)
            R1 = self.incep3(R1)
            if i == 0:
                ret = R1
            else:
                ret = torch.cat((ret,R1),dim=1)

        attentive2 = F.avg_pool2d(attentive,kernel_size=2,stride=2)
        for i in range(8):
            temp = attentive2[:,i].clone()
            temp = temp.view(-1,1,17,17).expand(-1,768,17,17)
            R2 = y*temp
            R2 = self.incep3_2(R2)
            ret = torch.cat((ret,R2),dim=1)

        attentive3 = F.avg_pool2d(attentive,kernel_size=4,stride=4)
        for i in range(8):
            temp = attentive3[:,i].clone()
            temp = temp.view(-1,1,8,8).expand(-1,2048,8,8)
            R3 = z*temp
            ret = torch.cat((ret,R3),dim=1)

        return ret

class AF2(nn.Module):
    def __init__(self):
        super(AF2,self).__init__()
        self.att = BasicConv2d(768,8,kernel_size=1)
        self.incep2 = nn.Sequential(InceptionB(288),InceptionC(768,channels_7x7=128),InceptionC(768,channels_7x7=160),InceptionC(768,channels_7x7=160),InceptionC(768,channels_7x7=192))
        self.incep3 = nn.Sequential(InceptionD(768),InceptionE(1280),InceptionE(2048))
        self.incep3_2 = nn.Sequential(InceptionD(768),InceptionE(1280),InceptionE(2048))
        self.patch = nn.ReflectionPad2d((0,1,0,1))
    def forward(self,input):
        x,y,z = input
        attentive = self.att(y)
        attentive1 = self.patch(F.upsample(attentive,scale_factor=2))
        for i in range(8):
            temp = attentive1[:,i].clone()
            temp = temp.view(-1,1,35,35).expand(-1,288,35,35)
            R1 = x *temp
            R1 = self.incep2(R1)
            R1 = self.incep3(R1)
            if i == 0:
                ret =R1
            else:
                ret = torch.cat((ret,R1),dim=1)

        attentive2 = attentive
        for i in range(8):
            temp = attentive2[:,i].clone()
            temp = temp.view(-1,1,17,17).expand(-1,768,17,17)
            R2 = y*temp
            R2 = self.incep3_2(R2)
            ret = torch.cat((ret,R2),dim=1)

        attentive3 = F.avg_pool2d(attentive,kernel_size=2,stride=2)
        for i in range(8):
            temp = attentive3[:,i].clone()
            temp = temp.view(-1,1,8,8).expand(-1,2048,8,8)
            R3 = z *temp
            ret = torch.cat((ret,R3),dim=1)

        return ret

class AF3(nn.Module):
    def __init__(self):
        super(AF3,self).__init__()
        self.att = BasicConv2d(2048,8,kernel_size=1)
        self.incep2 = nn.Sequential(InceptionB(288),InceptionC(768,channels_7x7=128),InceptionC(768,channels_7x7=160),InceptionC(768,channels_7x7=160),InceptionC(768,channels_7x7=192))
        self.incep3 = nn.Sequential(InceptionD(768),InceptionE(1280),InceptionE(2048))
        self.incep3_2 = nn.Sequential(InceptionD(768),InceptionE(1280),InceptionE(2048))
        self.patch = nn.ReflectionPad2d((0,1,0,1))
        self.patch2 = nn.ReflectionPad2d((0,1,0,1))

    def forward(self,input):
        x,y,z = input
        attentive = self.att(z)
        attentive2 = self.patch(F.upsample(attentive,scale_factor=2))
        attentive1 = self.patch(F.upsample(attentive2,scale_factor=2))
        for i in range(8):
            temp = attentive1[:,i].contiguous().view(-1,1,35,35).expand(-1,288,35,35)
            R1 = x *temp
            R1 = self.incep2(R1)
            R1 = self.incep3(R1)
            if i == 0:
                ret = R1
            else:
                ret = torch.cat((ret,R1),dim=1)

        for i in range(8):
            temp = attentive2[:,i].contiguous().view(-1,1,17,17).expand(-1,768,17,17)
            R2 = y*temp 
            R2 = self.incep3_2(R2)
            ret = torch.cat((ret,R2),dim=1)

        attentive3 = attentive
        for i in range(8):
            temp = attentive3[:,i].contiguous().view(-1,1,8,8).expand(-1,2048,8,8)
            R3 = z*temp
            ret = torch.cat((ret,R3),dim=1)


        return ret
