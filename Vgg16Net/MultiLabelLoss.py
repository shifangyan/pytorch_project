#!/usr/bin/env python
# coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as func
class MultiLabelLoss(nn.Module):
    def __init__(self):
        super(MultiLabelLoss,self).__init__()
        return

    def forward(self,input,target):
        batch_size =  input.size()[0]
        loss = func.binary_cross_entropy_with_logits(input,target,reduction="sum") #size_average 将被弃用 改为reduction="sum"  
        return loss/float(batch_size)




if __name__ == "__main__":
    target = torch.Tensor([[1,1,0],[0,1,1]])  #猫 狗 人
    input = torch.Tensor([[0,0,0],[0,0,0]])  

    multi_label_loss = MultiLabelLoss()
    loss = multi_label_loss(input,target)
    print loss
