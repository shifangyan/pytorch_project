#!/usr/bin/env python
# coding=utf-8

from torch.utils.data import Dataset
from torchvision import transforms
#from image_transforms import *
import skimage.io as io
import torch
#import matplotlib.pyplot as plt
import os

#################################################################
#此为我的分类数据集pytorch标准读取方式
#修改时间：2019.1.3
#修改人员：sfy
#################################################################
class GlassesDataset(Dataset):
    def __init__(self,img_list_path,dataset_root_path="/dataset/human_attribute/",transform=None):
        self.img_list_path = img_list_path
        self.transform = transform
        self.img_label_dic = {"img_path":[],"label":[]}
        self.dataset_root_path = dataset_root_path
        fr = open(self.img_list_path,"r")
        for line in fr:
            strings = line.split(" ")
            strings[1] = strings[1].split("\n")[0]
            self.img_label_dic["img_path"].append(strings[0])
            self.img_label_dic["label"].append(strings[1])

    def __len__(self):
        if(len(self.img_label_dic["img_path"]) != len(self.img_label_dic["label"])):
            return 0
        return len(self.img_label_dic["img_path"])

    def __getitem__(self,idx):
        img_path = os.path.join(self.dataset_root_path,self.img_label_dic["img_path"][idx])
        img = io.imread(img_path)
        label = int(self.img_label_dic["label"][idx]) #切记这一步要把label转换成int 否则后面不能转换成tensor
        if self.transform:
            img = self.transform(img)
        label = torch.LongTensor([label]) #要求必须是longtensor 不能是floattensor
        return img,label.squeeze()


##此为测试文件
if __name__ =="__main__":
    train_dataset = DogsDataset("/home/dataset/dogs-in-the-wild/annotation/train.txt",
                                 transforms.Compose([Resize(256),Flip(),RandomCrop(227,"train")]))
    for i in range(len(train_dataset)):
        image,label = train_dataset[i]
        if i == 10:
            print type(image)
            #print image.shape
            plt.figure("image")
            plt.imshow(image[10])
            #print image[0]
            plt.show()
