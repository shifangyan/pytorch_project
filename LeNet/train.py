#!/usr/bin/env python
# coding=utf-8

from torch.utils.data import DataLoader
import torch
from MNISTDataset import MNISTDataset
from LeNet import LeNet 
import skimage.io as io
from torch.autograd import Variable
import matplotlib.pyplot as plt

learning_rate = 0.0001
batch_size = 64  #60000/64=937.5
iter_display = 100 #迭代多少次显示一次
num_epoch = 10
test_iterval = 1000  #每训练多少次进行一次测试
#test_iter = 150  #测试进行多少次 10000/64 = 156 

train_dataset = MNISTDataset("./mnist/train.txt")
test_dataset = MNISTDataset("./mnist/test.txt")
train_loader = DataLoader(train_dataset,batch_size,True) #数据加载器  主要实现了批量加载 数据打乱 以及并行化读取数据 pytorch自带  也可以自己实现
test_loader = DataLoader(test_dataset,batch_size,True) 

lenet_model = LeNet() #实例化类LeNet  实际上也调用了LeNet的__init__()函数
lenet_model = lenet_model.cuda()  #转移到GPU上
print(lenet_model)

optimizer = torch.optim.SGD(lenet_model.parameters(),learning_rate,0.99) #实例化一个SGD优化器 参数：需要更新的参数  学习率 momentum(默认为0) 正则化项系数(默认为0) dampening(动量抑制因子 默认为0) nesterov(是否使用Nesterov 默认为false)
loss_func = torch.nn.CrossEntropyLoss()  #softmax loss函数

train_loss = 0.0
train_acc = 0.0
test_loss = 0.0
test_acc = 0.0
num_iter = -1
recoder = {"num_iter":[],"train_loss":[],"train_acc":[],"test_loss":[],"test_acc":[]}
for epoch in range(num_epoch):
    for index,(batch_img,batch_label) in enumerate(train_loader):
      #  print index
        num_iter += 1
      #  print type(batch_img)
      #  print type(batch_label)
        batch_img = Variable(batch_img.cuda())   #将数据拷贝到GPU上运行，并且放入到Variable中
        batch_label = Variable(batch_label.cuda())
        lenet_model.train()
        output = lenet_model(batch_img) #进行前向计算
      #  print output.data.size()  #output输出为batch_size*10 
      #  print batch_label.data.size()
        loss = loss_func(output,batch_label) #计算loss 此loss已经在一个batch_size上做了平均
        #print loss.data.size()  loss输出为一个标量
        train_loss += loss.item()  
        max,max_index = torch.max(output,1)  #返回1维度上的最大值 记忆下标
        #print max.size()
        train_correct = torch.sum((max_index.data == batch_label.data))  #统计一个batch中预测正确的数量
      #  print "train_correct:",train_correct
        train_acc += train_correct  
      #  print "train_acc:",train_acc
        #反向传播
        lenet_model.zero_grad()  #因为梯度是累计的 所以先将梯度清0
        loss.backward()  #将自动计算所有可学习参数的梯度
        optimizer.step()  #调用优化器来更新参数


        #显示设置
        if(num_iter % iter_display == 0):
            print "iter_num:",num_iter
            if num_iter == 0:
                train_loss = train_loss
                train_acc = float(train_acc)/batch_size
            else:
                train_loss = train_loss / iter_display
                train_acc = float(train_acc)/(iter_display*batch_size)
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
            for test_batch_img,test_batch_label in test_loader:
                test_batch_img = Variable(test_batch_img.cuda())
                test_batch_label = Variable(test_batch_label.cuda())
                lenet_model.eval() #转换成测试模式 仅仅只对dropout层和batchnorm层有影响
                output = lenet_model(test_batch_img)  #测试前向计算
                loss = loss_func(output,test_batch_label)
                test_loss += loss.item()
                max,max_index = torch.max(output,1)
                test_correct = torch.sum((max_index.data == test_batch_label.data))
                test_acc += test_correct
            test_loss = test_loss/len(test_dataset)*batch_size
            test_acc = float(test_acc)/float(len(test_dataset))
            recoder["test_loss"].append(test_loss)
            recoder["test_acc"].append(test_acc)
            print "test loss:",test_loss
            print "test acc:",test_acc

#存储权重
torch.save(lenet_model,"./weight/LeNet.weights")
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
