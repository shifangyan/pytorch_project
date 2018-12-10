#!/usr/bin/env python
# coding=utf-8

from torch.utils.data import Dataset
from torchvision import transforms
import skimage.io as io
import torch
import matplotlib.pyplot as plt

class RAPDataset(Dataset):
    def __init__(self,img_list_path,transform=None):
        self.img_list_path = img_list_path
        self.transform = transform
        self.img_label_dic = {"img_path":[],"label":[]}
        fr = open(self.img_list_path,"r")
        for line in fr:
            strings = line.split(" ")
            strings[-1] = strings[-1].split("\n")[0]
        #    print strings[1]
            self.img_label_dic["img_path"].append(strings[0])
            self.img_label_dic["label"].append(strings[1:])

    def __len__(self):
        if(len(self.img_label_dic["img_path"]) != len(self.img_label_dic["label"])):
            return 0
        return len(self.img_label_dic["img_path"])

    def __getitem__(self,idx):
        path = "/home/dataset/human_attribute/RAP/RAP_dataset/" + self.img_label_dic["img_path"][idx]
        img = io.imread(path)
        label = [int(i) for i in self.img_label_dic["label"][idx]] #切记这一步要把label转换成int 否则后面不能转换成tensor
        if self.transform:
            img = self.transform(img)
       # print len(imgs)
        #labels = np.zeros((imgs.shape[0],1))
        #labels[:] = label
        label = torch.FloatTensor([label]) #要求必须是longtensor 不能是floattensor
        #img = torch.Tensor(img)
        #img = img.view(1,img.size()[0],img.size()[1]) #添加上1维度
        #print img.shape,label.shape
        return img,label.squeeze()


if __name__ =="__main__":
    train_dataset = PA100KDataset("/home/dataset/human_attribute/PA100K/train.txt")
    for i in range(len(train_dataset)):
        image,label = train_dataset[i]
        print label.shape
            #print image.shape
        plt.figure("image")
        plt.imshow(image)
            #print image[0]
        plt.show()
