#!/usr/bin/env python
# coding=utf-8
#此文件实现一些检测当中用到的预处理
from torchvision import transforms
import random

class RandomHorizontalFlipWithBBox(object):
    #以一定概率进行水平翻转 同时翻转ground truth box的位置
    def __init__(self,p=0.5):
        self.p = p
        self.transform = transforms.RandomHorizontalFlip(p)
    def __call__(self,img,bbox): #bbox [[xmin,ymin,xmax,ymax]]
        if random.random() < self.p:
            img_w,img_h = img.size
            #print bbox
            for i in bbox:
                temp = i[0]
                i[0] = img_w - i[2] #new_x_min = img_w - x_max
                i[2] = img_w - temp
            #print bbox
            transform = transforms.RandomHorizontalFlip(1)
            img = transform(img)
            return img,bbox
        return img,bbox


class ResizeWithBBox(object):
    #改变图像大小，同时也改变ground truth box的大小
    #如果是元组类型，输出的图片尺寸与output_size匹配，按照(height,width)顺序
    #如果是整数类型，输出的图片按等比例缩放
    def __init__(self,output_size):
        assert isinstance(output_size,(tuple,int))
        self.output_size = output_size

    def __call__(self,img,bbox):
        img_w,img_h = img.size
        if isinstance(self.output_size,int):
            if h>w:
                new_h,new_w = self.output_size*h/w,self.output_size
            else:
                new_h,new_w = self.output_size,self.output_size*w/h
        else:
            new_h,new_w = self.output_size

        new_h,new_w = int(new_h),int(new_w)
        for i in bbox:
            i[0] = int(i[0]*new_w/img_w)
            i[1] = int(i[1]*new_h/img_h)
            i[2] = int(i[2]*new_w/img_w)
            i[3] = int(i[3]*new_h/img_h)
      #  print bbox
        transform = transforms.Resize(self.output_size)
        img = transform(img)
        return img,bbox

