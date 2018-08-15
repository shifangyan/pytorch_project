#!/usr/bin/env python
# coding=utf-8

from torch.utils.data import Dataset
import skimage.io as io
import torch

class MNISTDataset(Dataset):
    #mnist手写体数据集
    def __init__(self,img_list_path,transform=None):
        self.img_list_path = img_list_path
        self.transform = transform
        self.img_label_dic = {"img_path":[],"label":[]}
        fr = open(self.img_list_path,"r")
        for line in fr:
            strings = line.split(" ")
            strings[1] = strings[1].split("\n")[0]
        #    print strings[1]
            self.img_label_dic["img_path"].append(strings[0])
            self.img_label_dic["label"].append(strings[1])

    def __len__(self):
        if(len(self.img_label_dic["img_path"]) != len(self.img_label_dic["label"])):
            return 0
        return len(self.img_label_dic["img_path"])

    def __getitem__(self,idx):
        img = io.imread(self.img_label_dic["img_path"][idx])
        label = int(self.img_label_dic["label"][idx]) #切记这一步要把label转换成int 否则后面不能转换成tensor
        img = torch.Tensor(img)
        img = img.view(1,img.size()[0],img.size()[1]) #添加上1维度
        #print img.size(),label.size()
        return img,label
