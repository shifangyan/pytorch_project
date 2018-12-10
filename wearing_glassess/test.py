#!/usr/bin/env python
# coding=utf-8

from torch.utils.data import DataLoader
import torch
from GlassessDataset import GlassessDataset
from AlexNet import AlexNet
import skimage.io as io
import os
from image_transforms import *

batch_size = 128
image_path = "/home/dataset/mulvideo/9-24/face/"
load_path = "AlexNet_7000.weight"
img_save_path = "/home/dataset/human_attribute/glassess/prediction/"

#初始化alexnet模型
torch.cuda.set_device(1)
alex_model = AlexNet()
alex_model = alex_model.cuda()
print(alex_model)

#加载模型参数
checkpoint = torch.load(load_path)
pretrain_dict = checkpoint["net"]
model_dict = alex_model.state_dict()
pretrain_dict = {k:v for k,v in pretrain_dict.items() if k in model_dict}
model_dict.update(pretrain_dict)
alex_model.load_state_dict(model_dict)

#初始化图片预处理类
resize = Resize(256)
rand_crop = RandomCrop(227)
to_tensor = ToTensor()

dirlist = os.listdir(image_path)
#开始预测图片
num = 0
for img_path in dirlist:
    full_path = os.path.join(image_path,img_path) 
    src_img = io.imread(full_path)
    img = resize(src_img)
    img = rand_crop(img)
    img = to_tensor(img)
    img = img.contiguous().view(1,img.shape[0],img.shape[1],img.shape[2])
    img = Variable(img.cuda())
    output = alex_model(img)
    max,max_index = torch.max(output,1)
    #print max_index.data
    if max_index.data[0] == 0:  #不戴眼镜
        label = "0"
    elif max_index.data[0] == 1: #戴眼镜
        label = "1"
    full_save_path = os.path.join(os.path.join(img_save_path,label),img_path)
    io.imsave(full_save_path,src_img)
    num += 1
    print "success to predict",num,"images"
print "finish"
