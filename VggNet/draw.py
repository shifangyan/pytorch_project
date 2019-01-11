#!/usr/bin/env python
# coding=utf-8
import matplotlib.pyplot as plt

test_iterval = 1000
recoder = {"num_iter":[],"train_loss":[],"test_loss":[],"train_acc":[],"test_acc":[]}
log_file = "./201812171359.log"
log_fd = open(log_file,"r")
find_log = 0
for line in log_fd.readlines():
#int begin_index = 0
    if(line.find("iter_num")!= -1):
        recoder["num_iter"].append(int(line[line.find("iter_num")+9:-1]))
    if(line.find("train loss") != -1):
        recoder["train_loss"].append(float(line[line.find("train loss")+11:-1]))
    if(line.find("train acc") != -1):
        recoder["train_acc"].append(float(line[line.find("train acc")+10:-1]))
    if(line.find("test loss") != -1):
        recoder["test_loss"].append(float(line[line.find("test loss")+10:-1]))
    if(line.find("test acc") != -1):
        recoder["test_acc"].append(float(line[line.find("test acc")+9:-1]))

print recoder

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

