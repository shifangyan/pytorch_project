#!/usr/bin/env python
# coding=utf-8
import matplotlib.pyplot as plt

test_iterval = 1000
recoder1 = {"num_iter":[],"train_loss":[],"test_loss":[],"train_acc":[],"test_acc":[]}
recoder2 = {"num_iter":[],"train_loss":[],"test_loss":[],"train_acc":[],"test_acc":[]}
log_file = "./201901041014.log"
log_fd = open(log_file,"r")
for line in log_fd.readlines():
#int begin_index = 0
    if(line.find("iter_num")!= -1):
        recoder1["num_iter"].append(int(line[line.find("iter_num")+9:-1]))
    if(line.find("train loss") != -1):
        recoder1["train_loss"].append(float(line[line.find("train loss")+11:-1]))
    if(line.find("train acc") != -1):
        recoder1["train_acc"].append(1-float(line[line.find("train acc")+10:-1]))
    if(line.find("test loss") != -1):
        recoder1["test_loss"].append(float(line[line.find("test loss")+10:-1]))
    if(line.find("test acc") != -1):
        recoder1["test_acc"].append(1-float(line[line.find("test acc")+9:-1]))

log_file = "./201901041015.log"
log_fd = open(log_file,"r")
for line in log_fd.readlines():
    if(line.find("iter_num")!= -1):
        recoder2["num_iter"].append(int(line[line.find("iter_num")+9:-1]))
    if(line.find("train loss") != -1):
        recoder2["train_loss"].append(float(line[line.find("train loss")+11:-1]))
    if(line.find("train acc") != -1):
        recoder2["train_acc"].append(1-float(line[line.find("train acc")+10:-1]))
    if(line.find("test loss") != -1):
        recoder2["test_loss"].append(float(line[line.find("test loss")+10:-1]))
    if(line.find("test acc") != -1):
        recoder2["test_acc"].append(1-float(line[line.find("test acc")+9:-1]))
    
#print recoder

#绘图
plt.figure("loss")
plt.plot(recoder1["num_iter"],recoder1["train_loss"],color = "blue",linewidth = 1,label = "resnet18-train-loss")
x1 =  []
x2 = []
for iter in recoder1["num_iter"]:
    if iter % test_iterval ==0:
        x1.append(iter)
for iter in recoder2["num_iter"]:
    if iter %test_iterval == 0:
        x2.append(iter)
print x1
plt.plot(x1,recoder1["test_loss"],color = "blue",linewidth = 3,label = "resnet18-test-loss")
plt.plot(recoder2["num_iter"],recoder2["train_loss"],color = "red",linewidth = 1,label = "resnet34-train-loss")
plt.plot(x2,recoder2["test_loss"],color = "red",linewidth = 3,label = "resnet34-test-loss")

plt.figure("err")
plt.plot(recoder1["num_iter"],recoder1["train_acc"],color = "blue",linewidth = 1,label = "resnet18-train-error")
plt.plot(x1,recoder1["test_acc"],color = "blue",linewidth = 3,label = "resnet18-test-error")
plt.plot(recoder2["num_iter"],recoder2["train_acc"],color = "red",linewidth = 1,label = "resnet34-train-error")
plt.plot(x2,recoder2["test_acc"],color = "red",linewidth = 3,label = "resnet34-test-error")
plt.legend(loc="upper left")
plt.xlim((10000,13000))
plt.ylim((0.0,0.15))
plt.show()

