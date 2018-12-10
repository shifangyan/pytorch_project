#!/usr/bin/env python
# coding=utf-8

from torch.utils.data import DataLoader
import torch
#from DogsDataset import DogsDataset
from GlassesDataset import GlassesDataset
from AlexNet import AlexNet 
import skimage.io as io
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torchvision import transforms
from image_transforms import *
from torchvision import models
from torch import nn

learning_rate = 0.001
train_batch_size = 64  #60000/64=937.5
test_batch_size = 16
image_rate = 1  #每张图片经过预处理扩展的倍数
iter_display = 100 #迭代多少次显示一次
num_epoch = 250
test_iterval = 500  #每训练多少次进行一次测试
iter_save = 500 #美巡多少次进行一次参数保存  可用于断点训练
save_path = "./weight/AlexNet"   #参数保存路径 会在后面加后缀 例如 ./weight/AlexNet_1000.weight
load_path = "./weight/AlexNet_8000.weight"  #参数读取路径
use_save_parameter = False  #决定是否使用保存参数
checkpoint_train = False  #决定是否进行断点训练
#test_iter = 150  #测试进行多少次 10000/64 = 156 

torch.cuda.set_device(0)
train_dataset = GlassesDataset("/home/dataset/human_attribute/glassess/train.txt",
                            transform = transforms.Compose([transforms.ToPILImage(),transforms.ColorJitter(0.05,0.05,0.05,0.05),transforms.RandomHorizontalFlip(),transforms.Resize(256),transforms.RandomCrop((227,227)),transforms.ToTensor()]))
test_dataset = GlassesDataset("/home/dataset/human_attribute/glassess/test.txt",
                              transform = transforms.Compose([transforms.ToPILImage(),transforms.Resize(256),transforms.FiveCrop((227,227)),transforms.Lambda(lambda crops:torch.stack([transforms.ToTensor()(crop) for crop in crops]))]))
train_loader = DataLoader(train_dataset,train_batch_size,True) #数据加载器  主要实现了批量加载 数据打乱 以及并行化读取数据 pytorch自带  也可以自己实现
test_loader = DataLoader(test_dataset,test_batch_size,True) 

#alex_model = AlexNet() #实例化类LeNet  实际上也调用了LeNet的__init__()函数
#alex_model.initialize_parameters()
alex_model = models.alexnet(pretrained = True)  #使用pytorch的预训练模型
alex_model.classifier = nn.Sequential(
                                     nn.Linear(9216,4096),
                                     nn.ReLU(),
                                  #   nn.Dropout(0.5),
                                     nn.Linear(4096,4096),
                                     nn.ReLU(),
                                     nn.Linear(4096,2)) 
alex_model = alex_model.cuda()  #转移到GPU上
print(alex_model)

optimizer = torch.optim.SGD(alex_model.parameters(),learning_rate,0.99,0.0005) #实例化一个SGD优化器 参数：需要更新的参数  学习率 momentum(默认为0) 正则化项系数(默认为0) dampening(动量抑制因子 默认为0) nesterov(是否使用Nesterov 默认为false)
loss_func = torch.nn.CrossEntropyLoss()  #softmax loss函数
start_epoch = 0
#读取断点训练参数
if use_save_parameter == True:
    checkpoint = torch.load(load_path)
    pretrain_dict = checkpoint["net"]
    model_dict = alex_model.state_dict()
    pretrain_dict = {k:v for k,v in pretrain_dict.items() if k in model_dict}
    model_dict.update(pretrain_dict)
    alex_model.load_state_dict(model_dict)
   # alex_model.load_state_dict(checkpoint["net"])
    if checkpoint_train == True:
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch =  checkpoint["epoch"] +1
    print "success to load parameter"
train_loss = 0.0
train_acc = 0.0
test_loss = 0.0
test_acc = 0.0
num_iter = 0
recoder = {"num_iter":[],"train_loss":[],"train_acc":[],"test_loss":[],"test_acc":[]}

print "start_epoch:",start_epoch
for epoch in range(start_epoch,num_epoch):
    for batch_img,batch_label in train_loader:
        num_iter += 1
      #  print type(batch_img)
      #  print type(batch_label)
      #  batch_img = batch_img.view(-1,batch_img.shape[2],batch_img.shape[3],batch_img.shape[4])
      #  batch_label = batch_label.view(batch_label.shape[0]*batch_label.shape[1])
      #  batch_label = batch_label.view(batch_label.shape[0])
      #  print batch_img.shape

      #  print batch_label.shape
        batch_img = Variable(batch_img.cuda())   #将数据拷贝到GPU上运行，并且放入到Variable中
        batch_label = Variable(batch_label.cuda())
        alex_model.train()
        output = alex_model(batch_img) #进行前向计算
      #  print output.data.size()  #output输出为batch_size*10 
      #  print batch_label.data.size()
        loss = loss_func(output,batch_label) #计算loss 此loss已经在一个batch_size上做了平均
        #print loss.data.size()  loss输出为一个标量
      #  train_loss += loss.item()  
        train_loss += loss.item()
        max,max_index = torch.max(output,1)  #返回1维度上的最大值 记忆下标
        #print max.size()
        train_correct = torch.sum((max_index.data == batch_label.data))  #统计一个batch中预测正确的数量
      #  print "train_correct:",train_correct
        train_acc += train_correct  
      #  print "train_acc:",train_acc
        #反向传播
        alex_model.zero_grad()  #因为梯度是累计的 所以先将梯度清0
        loss.backward()  #将自动计算所有可学习参数的梯度
        optimizer.step()  #调用优化器来更新参数


        #显示设置
        if(num_iter % iter_display == 0 or num_iter == 1):
            print "iter_num:",num_iter
            if num_iter == 1:
                k = 1
            else:
                k = iter_display
            train_loss = train_loss/k
            train_acc = float(train_acc)/float(k*train_batch_size*image_rate)
            recoder["num_iter"].append(num_iter)
            recoder["train_loss"].append(train_loss)
            recoder["train_acc"].append(train_acc)
            print "train loss:",train_loss
            print "train acc:",train_acc
            train_loss = 0
            train_acc = 0
        
        #测试
        if(num_iter % test_iterval == 0):
            test_loss = 0
            test_acc = 0
            test_num_iter = 0
            alex_model.eval()
            for test_batch_img,test_batch_label in test_loader:
               # print test_batch_img.size()
                test_num_iter  += 1
                test_batch_img = Variable(test_batch_img.cuda())
                test_batch_label = Variable(test_batch_label.cuda())
                bs,ncrops,c,h,w = test_batch_img.size()
                output = alex_model(test_batch_img.view(-1,c,h,w))  #测试前向计算
                output_avg = output.view(bs,ncrops,-1).mean(1)  
                loss = loss_func(output_avg,test_batch_label)
                test_loss += loss.item()
               # test_loss += loss.data[0]
              #  print test_batch_label.data
                max,max_index = torch.max(output_avg,1)
                test_correct = torch.sum((max_index.data == test_batch_label.data))
                test_acc += test_correct
            test_loss = test_loss/test_num_iter
            test_acc = float(test_acc)/float(len(test_dataset))
            recoder["test_loss"].append(test_loss)
            recoder["test_acc"].append(test_acc)
            print "test loss:",test_loss
            print "test acc:",test_acc

        #参数保存
        if(num_iter % iter_save == 0):
            state = {"net":alex_model.state_dict(),"optimizer":optimizer.state_dict(),"epoch":epoch}
            full_save_path = save_path +"_" + str(num_iter) +".weight"
            torch.save(state,full_save_path)
            print "success to save parameter"
#训练完成 再进行一次存储权重
state = {"net":alex_model.state_dict(),"optimizer":optimizer.state_dict(),"epoch":epoch}
full_save_path = save_path +"_" + str(num_iter) +".weight"
torch.save(state,full_save_path)
#绘图
plt.figure("loss")
plt.plot(recoder["num_iter"],recoder["train_loss"])
x = []
for iter in recoder["num_iter"]:
    if iter % test_iterval ==0:
        x.append(iter)
plt.plot(x,recoder["test_loss"])

plt.figure("acc")
plt.plot(recoder["num_iter"],recoder["train_acc"])
plt.plot(x,recoder["test_acc"])
plt.show()
