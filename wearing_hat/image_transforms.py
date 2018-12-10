#!/usr/bin/env python
# coding=utf-8
#此文件实现图片预处理的一些类
from skimage import transform
#import PIL.image as img
import numpy as np
import torch

class Resize(object):
    # output_size (tuple or int) 如果是元组类型，输出的图片尺寸与output_size匹配 按照(height,width)顺序
    #                            如果是整数类型，输出的图片的短边与output_size匹配，图片等比例缩放
    def __init__(self,output_size):
        assert isinstance(output_size,(tuple,int))
        self.output_size = output_size

    def __call__(self,image):
        #print "Resize:",image.shape
        h,w = image.shape[:2] #numpy格式的图片维度为(h,w,c)
        if isinstance(self.output_size,int):
            if h>w:
                new_h,new_w = self.output_size*h/w,self.output_size
            else:
                new_h,new_w = self.output_size,self.output_size*w/h
        else:
            new_h,new_w = self.output_size

        new_h,new_w = int(new_h),int(new_w)
        image = transform.resize(image,(new_h,new_w)) #调用的skimage.transform
        #print image.shape
        return image

class RandomCrop(object):
    #随机裁剪图片 训练时 每张图片随机裁剪出5张  测试时 进行中心裁剪
    #output_size 期望输出的大小 (tuple or int) 如果是tuple 按照(h,w)裁剪
    #如果是int 裁剪为正方形
    def __init__(self,output_size,mode = "test"):
        assert isinstance(output_size,(tuple,int))
        self.output_size = output_size
        self.mode = mode
        if self.mode == "train":
            self.num_crop = 5
        else:
            self.num_crop = 1
    def __call__(self,imgs):
        if isinstance(self.output_size,int):
            new_h,new_w = self.output_size,self.output_size
        else:
            new_h,new_w = self.output_size
        if self.mode == "train":
            crop_imgs = np.zeros((self.num_crop*imgs.shape[0],new_h,new_w,imgs.shape[3]))
        else:
            crop_imgs = np.zeros((self.num_crop,new_h,new_w,imgs.shape[2]))
            #print crop_imgs.shape
       # print crop_imgs.shape
        if self.mode =="train":
            index = -1
            for img in imgs:
                h,w = img.shape[:2]
                for i in range(self.num_crop):
                    index += 1
                    left = np.random.randint(0,w-new_w) #确定裁剪的左上角x坐标
                    top = np.random.randint(0,h-new_h)  #确定裁剪的左上角y坐标
                    image = img[top:top+new_h,left:left+new_w]
                #    print image.shape
                    crop_imgs[index] = image
        
            return crop_imgs
        else:
            h,w = imgs.shape[:2]
            left = (w-new_w)/2
            top = (h-new_h)/2
            image = imgs[top:top+new_h,left:left+new_w]
            return image
        

class Flip(object):
    #水平和竖直翻转图片 输入：原始图片  输出：[原始图片，左右翻转图片,上下翻转图片]
    def __init__(self):
        return

    def __call__(self,image):
        #print image.shape
        h,w,c = image.shape
        h_img = np.zeros_like(image)
        v_img = np.zeros_like(image)
        imgs = np.zeros((3,h,w,c))
        for i in range(h):
            for j in range(w):
       #         print i,j
                h_img[i,w-1-j]= image[i,j]
                v_img[h-1-i,j] = image[i,j]
        imgs[0] = image
        imgs[1] = h_img
        imgs[2] = v_img
        #print imgs.shape
        return imgs

class ToTensor(object):
    #将numpy格式的图片，即ndarray格式的转换成pytorch的tensor格式
    def __init__(self,mode="test"):
        self.mode = mode
        return

    def __call__(self,imgs):
        #交换颜色通道
        #numpy(ndarray)图片：H*W*C
        #pytorch(tensor)图片：C*H*W
        if self.mode == "train":
            imgs = imgs.transpose((0,3,1,2)) 
            imgs = torch.Tensor(imgs)
        else:
            imgs = imgs.transpose((2,0,1))
            imgs = torch.Tensor(imgs)
        return imgs
