#!/usr/bin/env python
# coding=utf-8

import torch
import torch.nn as nn
from VocDataset import *

class YoloV1Loss(nn.Module):
    grid_num = 7
    bounding_box = 2
    class_num = 20
    img_size = 448
    coord = 5
    noobj = 0.05
    grid_size = img_size/grid_num
    def __init__(self):
        super(YoloV1Loss,self).__init__()
        self.loss_func = torch.nn.MSELoss(reduction = "sum")
    def compute_iou(self,pred,target):
        iou = torch.zeros(target.size()[0],target.size()[1],target.size()[2],2)

        for i in range(target.size()[0]):
            for j in range(self.grid_num):
                for k in range(self.grid_num):
                    if target[i,j,k,4] == 1: #存在ground truth box
                        x_min_target = (target[i,j,k,0]+k)*self.grid_size-0.5*target[i,j,k,2]*self.img_size
                        y_min_target = (target[i,j,k,1]+j)*self.grid_size-0.5*target[i,j,k,3]*self.img_size
                        x_max_target = (target[i,j,k,0]+k)*self.grid_size+0.5*target[i,j,k,2]*self.img_size
                        y_max_target = (target[i,j,k,1]+j)*self.grid_size+0.5*target[i,j,k,3]*self.img_size
                        for t in range(2):
                        #计算xmin,ymin,xmax,ymax
                            x_min_pred = (pred[i,j,k,5*t]+k)*self.grid_size-0.5*pred[i,j,k,5*t+2]*self.img_size
                            y_min_pred = (pred[i,j,k,5*t+1]+j)*self.grid_size-0.5*pred[i,j,k,5*t+3]*self.img_size
                            x_max_pred = (pred[i,j,k,5*t]+k)*self.grid_size+0.5*pred[i,j,k,5*t+2]*self.img_size
                            y_max_pred = (pred[i,j,k,5*t+1]+j)*self.grid_size+0.5*pred[i,j,k,5*t+3]*self.img_size


                        #计算IOU
                            x_min_max = max(x_min_pred,x_min_target)
                            y_min_max = max(y_min_pred,y_min_target)
                            x_max_min = min(x_max_pred,x_max_target)
                            y_max_min = min(y_max_pred,y_max_target)

                            box_pred_area = (x_max_pred-x_min_pred)*(y_max_pred-y_min_pred)
                        #可能预测出的数据不符合矩形条件 
                            if box_pred_area <0:
                                #print "????????"
                                box_pred_area = 0
                            box_target_area = (x_max_target-x_min_target)*(y_max_target-y_min_target)
                            if box_target_area <= 0:
                                print "标签打错了"
                                box_target_area = 0
                        #重叠区域面积
                            inter_area = max(x_max_min-x_min_max,0)*max(y_max_min-y_min_max,0)
                        #IOU计算
                            iou[i,j,k,t] = float(inter_area)/float(box_pred_area+box_target_area-inter_area)

        return iou

    def forward(self,pred,target):
        batch_size = target.size()[0]
        #x_and_y_target = torch.zeros(batch_size,grid_num,grid_num,2)
        #x_and_y_pred = torch.zeros(batch_size,grid_num,grid_num,2)
        pred = pred.view(pred.size()[0],7,7,30)
        obj_mask = (target[:,:,:,4] == 1) #N*7*7
        noobj_mask = (target[:,:,:,4] == 0)
        obj_mask_temp = obj_mask.unsqueeze(-1).expand_as(target) #N*7*7*30
        #print obj_mask
        noobj_mask = noobj_mask.unsqueeze(-1).expand_as(target)
        obj_target_grid = target.masked_select(obj_mask_temp).view(-1,30)
        obj_pred_grid = pred.masked_select(obj_mask_temp).view(-1,30)
        noobj_target_grid = target.masked_select(noobj_mask).view(-1,30)
        noobj_pred_grid = pred.masked_select(noobj_mask).view(-1,30)
        target_classses = obj_target_grid[:,10:]
        pred_classes = obj_pred_grid[:,10:]

        noobj_target_predictor = noobj_target_grid[:,:10].contiguous().view(-1,5)
        noobj_pred_predictor = noobj_pred_grid[:,:10].contiguous().view(-1,5)
        iou = self.compute_iou(pred,target).cuda()
        #print iou
        _,iou_max_index = torch.max(iou,3)

        #print max_index.size()
        #print max_index
        obj_target_predictor = torch.zeros(obj_target_grid.size()[0],5).cuda()
        obj_pred_predictor = torch.zeros(obj_pred_grid.size()[0],5).cuda()
        t = 0
        for i in range(target.size()[0]):
            for j in range(self.grid_num):
                for k in range(self.grid_num):
                    if target[i,j,k,4] == 1 and iou_max_index[i,j,k] == 0: 
                        obj_target_predictor[t] = target[i,j,k,0:5]
                        obj_pred_predictor[t] = pred[i,j,k,0:5]
                        t +=1
                        target[i,j,k,5:10] = 0
                        noobj_target_predictor += target[i,j,k,5:10]
                        noobj_pred_predictor += pred[i,j,k,5:10]
                    elif target[i,j,k,4] == 1 and iou_max_index[i,j,k] == 1:
                        obj_target_predictor[t] = target[i,j,k,5:10]
                        obj_pred_predictor[t] = pred[i,j,k,5:10]
                        t +=1 
                        target[i,j,k,0:5] = 0
                        #print target[i,j,k,0:5].unsqueeze(0).size()
                        noobj_target_predictor = torch.cat((noobj_target_predictor,target[i,j,k,0:5].unsqueeze(0)),0)
                        noobj_pred_predictor = torch.cat((noobj_pred_predictor,pred[i,j,k,0:5].unsqueeze(0)),0)
       #1 计算有目标的预测器 x y 的loss
        loss1 = self.coord*self.loss_func(obj_pred_predictor[:,0:2],obj_target_predictor[:,0:2])
       #2 计算有目标的预测器 w h 的loss
        obj_pred_location = nn.functional.relu(obj_pred_predictor[:,2:4])
        obj_pred_location = torch.sqrt(obj_pred_location)
        obj_target_location = torch.sqrt(obj_target_predictor[:,2:4])

        loss2 = self.coord*self.loss_func(obj_pred_location,obj_target_location)
       #3 计算有目标的预测器置信度 loss
        loss3 = self.loss_func(obj_pred_predictor[:,4],obj_target_predictor[:,4])
       #4 计算没有目标的预测器 置信度loss
        #print noobj_pred_predictor[:,4]
        #print noobj_target_predictor[:,4]
        loss4 = self.noobj * self.loss_func(noobj_pred_predictor[:,4],noobj_target_predictor[:,4])
       #5 计算有目标的grid 分类loss
        loss5 = self.loss_func(pred_classes,target_classses)
        loss = (loss1 + loss2 + loss3 + loss4 + loss5)/batch_size
        #print loss1,loss2,loss3,loss4,loss5
        return loss
        #print result.size()

def TestImgDisplay(img,pred):
    obj_iou_threshold = 0.5
    obj_confidence_threshold = 0.1
    pred = pred.view(-1,7,7,30)
    obj = []
    print pred[:,:,:,4]
    print pred[:,:,:,9]
    for i in range(7):
        for j in range(7):
            if pred[0,i,j,4] > obj_iou_threshold:
                classes_scroes_max,max_index = torch.max(pred[0,i,j,10:],0)
                obj_confidence = (pred[0,i,j,4] * classes_scroes_max).item()
                if obj_confidence > obj_confidence_threshold:
                    print pred[0,i,j,0:5]
if __name__ == "__main__":
    train_dataset = VocDataset("/dataset/voc2007/train.txt","/dataset",True)
    img,label = train_dataset[998]
    target = label.unsqueeze(0)
    pred = target.clone()
    for i in range(target.size()[0]):
        for j in range(7):
            for k in range(7):
                if target[i,j,k,4] == 1:
                    pred[i,j,k,2] = pred[i,j,k,2] * 0.5
                    pred[i,j,k,7] = pred[i,j,k,7] * 0.8

    yolo_loss = YoloV1Loss()
    iou = yolo_loss.compute_iou(pred,target)
    print iou
    yolo_loss.forward(pred,target)




