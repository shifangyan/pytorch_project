#!/usr/bin/env python
# coding=utf-8

from torch.utils.data import Dataset
from torchvision import transforms
from torchvision import utils
from image_transform import *
import skimage.io as io
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import os
import numpy as np

#################################################################
#此为我的分类数据集pytorch标准读取方式
#修改时间：2019.1.3
#修改人员：sfy
#################################################################
class VocDataset(Dataset):
    grid_num = 7
    bounding_box_num = 2
    class_num = 20
    def __init__(self,img_list_path,dataset_root_path="/dataset/human_attribute/",transform=None):
        self.img_list_path = img_list_path
        self.transform = transform
        self.img_label_dic = {"img_path":[],"label":[]}
        self.dataset_root_path = dataset_root_path
        self.voc2007_voc2012 = False
        if isinstance(img_list_path,list) and len(img_list_path) == 2:  #voc2007 + voc2012
            self.voc2007_voc2012 = True
        if self.voc2007_voc2012:
            for element in self.img_list_path:
                fr = open(element,"r")
                for line in fr:
                    strings = line.strip().split(" ")
                    self.img_label_dic["img_path"].append(strings[0])
                    self.img_label_dic["label"].append(strings[1:])
        else:
            fr = open(self.img_list_path,"r")
            for line in fr:
                strings = line.strip().split(" ")
                self.img_label_dic["img_path"].append(strings[0])
                self.img_label_dic["label"].append(strings[1:])

    def __len__(self):
        if(len(self.img_label_dic["img_path"]) != len(self.img_label_dic["label"])):
            return 0
        return len(self.img_label_dic["img_path"])

    def __getitem__(self,idx):
        img_path = os.path.join(self.dataset_root_path,self.img_label_dic["img_path"][idx])
        img = io.imread(img_path)
        #标签读取
        boxes = []
        classes = []
        strings = self.img_label_dic["label"][idx]
        for i in range(0,len(strings),5):
            boxes.append([int(strings[i]),int(strings[i+1]),int(strings[i+2]),int(strings[i+3])]) #[[x_min,y_min,x_max,y_max]]
            classes.append(int(strings[i+4]))
        #数据预处理
        if self.transform:
            transform = transforms.ToPILImage()
            img = transform(img)
            transform = transforms.ColorJitter(0.05,0.05,0.05,0.05)
            img = transform(img)
            transform = RandomHorizontalFlipWithBBox()
            img,boxes = transform(img,boxes)
            transform = ResizeWithBBox((448,448))
            img,boxes = transform(img,boxes)
            transform = transforms.ToTensor()
            img = transform(img)
        else:
            transform = transforms.ToPILImage()
            img = transform(img)
            transform = transforms.Resize((448,448))
            img = transform(img)
            transform = transforms.ToTensor()
            img = transform(img)

        #将标签转换成yolo格式
        #global grid_num
        #global bounding_box_num
        #global class_num
        label = torch.zeros(self.grid_num,self.grid_num,self.bounding_box_num*5+self.class_num)  #yolo的标签比较特殊 详情见论文
        for i ,element in enumerate(boxes):
            x_center = (element[0] +element[2])/2
            y_center = (element[1] + element[3])/2
            box_w = element[2] - element[0]
            box_h = element[3] - element[1]
            #print x_center,y_center,box_w,box_h
            class_label = classes[i] 
            #确认box在哪个grid里
            _,img_h,img_w = img.size()
            grid_index_x = float(self.grid_num*x_center)/float(img_w) #确认由哪个grid负责预测这个目标
            grid_index_y = float(self.grid_num*y_center)/float(img_h)  #确认由哪个grid负责预测这个目标
            offset_x = grid_index_x - int(grid_index_x)  #ground truth box的中心点 相对于负责预测这个box的grid左上角偏移量 0-1之间
            offset_y = grid_index_y - int(grid_index_y)
            relative_box_w = float(box_w)/float(img_w) #groud truth box的宽度和高度相对于图像宽度和高度的归一化值
            relative_box_h = float(box_h)/float(img_h)
            #填入标签tensor
            if label[int(grid_index_y),int(grid_index_x),4] == 1: #这就是一个grid对应两个目标的情况 yolo v1似乎忽略了这个问题
                label[int(grid_index_y),int(grid_index_x),:] = 0 #这里用第二个目标覆盖第一个目标
                #print "222"
            label[int(grid_index_y),int(grid_index_x),0] = offset_x
            label[int(grid_index_y),int(grid_index_x),5] = offset_x
            label[int(grid_index_y),int(grid_index_x),1] = offset_y
            label[int(grid_index_y),int(grid_index_x),6] = offset_y
            label[int(grid_index_y),int(grid_index_x),2] = relative_box_w
            label[int(grid_index_y),int(grid_index_x),7] = relative_box_w
            label[int(grid_index_y),int(grid_index_x),3] = relative_box_h
            label[int(grid_index_y),int(grid_index_x),8] = relative_box_h
            label[int(grid_index_y),int(grid_index_x),4] = 1  #训练时代表这个grid负责预测该目标
            label[int(grid_index_y),int(grid_index_x),9] = 1
            label[int(grid_index_y),int(grid_index_x),10+class_label] = 1  #代表该目标的类别
            
        return img,label


##此为测试文件
if __name__ =="__main__":
    train_dataset = VocDataset("/dataset/voc2007/train.txt","/dataset",True)
    img,label = train_dataset[998]
    bbox = []
    _,img_h,img_w = img.size()
    #print label
    for i in range(7):
        for j in range(7):
            if label[i,j,4] == 1:
                print "11"
                x_center = (j + label[i,j,0])/7.0 * img_w 
                y_center = (i+label[i,j,1])/7.0 * img_h
                w = label[i,j,2] * img_w 
                h = label[i,j,3] * img_h
                x_min = x_center - w/2
                y_min = y_center - h/2
                bbox.append([x_min,y_min,w,h])
    #print "ok"
    print len(bbox)
    img = utils.make_grid(img).numpy()
    fig = plt.figure("image")
    ax1 = fig.add_subplot(121)
    plt.imshow(np.transpose(img,(1,2,0)))
            #print image.shape
    ax2 = fig.add_subplot(122)
    plt.imshow(np.transpose(img,(1,2,0)))
    codes = [Path.MOVETO] + [Path.LINETO]*3 + [Path.CLOSEPOLY]
    for i in bbox:
        vertices = [(i[0],i[1]),(i[0]+i[2],i[1]),(i[0]+i[2],i[1]+i[3]),(i[0],i[1]+i[3]),(0,0)]
        #print vertices
        vertices = np.array(vertices,float)
        path = Path(vertices,codes)
        pathpatch = PathPatch(path,facecolor="None",edgecolor="red")
        plt.gca().add_patch(pathpatch)
    #plt.gca().add_patch(plt.Rectangle((355,32),50,91))
    plt.show()
