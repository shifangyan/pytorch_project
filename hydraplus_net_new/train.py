#!/usr/bin/env python
# coding=utf-8

from torch.utils.data import DataLoader
import torch
#from DogsDataset import DogsDataset
from RAPDataset import RAPDataset
from GlassesDataset import GlassesDataset
from InceptionV3 import * 
from HydraPlusNet import *
import skimage.io as io
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision import models
from torch import nn
import numpy as np
import logging 
import os.path
import time

learning_rate = 0.0001
num_class = 51
train_batch_size = 32  #60000/64=937.5
test_batch_size = 8
image_rate = 1  #每张图片经过预处理扩展的倍数
iter_display = 100 #迭代多少次显示一次
num_epoch = 200
is_test = True  #决定是否进行训练过程中的测试
test_iterval = 500  #每训练多少次进行一次测试
iter_save = 500 #美巡多少次进行一次参数保存  可用于断点训练
save_path = "./weight/InceptionV3"   #参数保存路径 会在后面加后缀 例如 ./weight/AlexNet_1000.weight
load_path = "./weight/InceptionV3_24000.weight"  #参数读取路径
use_save_parameter = False  #决定是否使用保存参数
checkpoint_train = False  #决定是否进行断点训练
#test_iter = 150  #测试进行多少次 10000/64 = 156 
#label_name = ["Female","AgeOver60","Age18-60","AgeLess18","Front","Side",       #PA100K
#              "Back","Hat","Glasses","HandBag","ShoulderBag","Backpack","HoldObjectsInFront",
#              "ShortSleeve","LongSleeve","UpperStride","UpperLogo","UpperPlaid","UpperSplice",
#              "LowerStripe","LowerPattern","LongCoat","Trousers","Shorts","Skirt&Dress","boots"]
#label_name = ["personalLess30","personalLess45","personalLess60","personalLarger60","carryingBackpack", #PETA
#              "carryingOther","lowerBodyCasual","upperBodyCasual","lowerBodyFormal","upperBodyFormal",
#              "accessoryHat","upperBodyJacket","lowerBodyJeans","footwearLeatherShoes","upperBodyLogo",
#              "hairLong","personalMale","carryingMessengerBag","accessoryMuffler","accessoryNothing",
#              "carryingNothing","upperBodyPlaid","carryingPlasticBags","footwearSandals","footwearShoes",
#              "lowerBodyShorts","upperBodyShortSleeve","lowerBodyShortSkirt","footwearSneaker","upperBodyThinStripes",
#              "accessorySunglasses","lowerBodyTrousers","upperBodyTshirt","upperBodyOther","upperBodyVNeck"]
label_name = ["Female","AgeLess16","Age17-30","Age31-45","BodyFat","BodyNormal","BodyThin","Customer","Clerk",  #RAP
              "BaldHead","LongHair","BlackHair","Hat","Glasses","Muffler","Shirt","Sweater","Vest","TShirt",
              "Cotton","Jacket","Suit-Up","Tight","ShortSleeve","LongTrousers","Skirt","ShortSkirt","Dress",
              "Jeans","TightTrousers","LeatherShoes","SportShoes","Boots","ClothShoes","CasualShoes",
              "Backpack","SSBag","HandBag","Box","PlasticBag","PaperBag","HandTrunk","OtherAttchment",
              "Calling","Talking","Gathering","Holding","Pusing","Pulling","CarryingbyArm","CarryingbyHand"]
#日志设置
#第一步 创建一个logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

#第二步 创建两个handler 一个写入日志文件 一个打印屏幕
rq = time.strftime("%Y%m%d%H%M",time.localtime(time.time()))
log_name = "./" + rq + ".log"
file_handler = logging.FileHandler(log_name,mode= "w")
file_handler.setLevel(logging.INFO)

print_handler = logging.StreamHandler()
print_handler.setLevel(logging.DEBUG)

#第三步 设置两个handler的输出格式
file_formatter = logging.Formatter("%(asctime)s-%(filename)s[line:%(lineno)d] - %(levelname)s:%(message)s")
file_handler.setFormatter(file_formatter)

#第四步 将handler添加到logger里
logger.addHandler(file_handler)
logger.addHandler(print_handler)


#logger.debug("debug")
#logger.info("info")
#logger.warning("warning")
#logger.error("error")
#logger.critical("critical")

torch.cuda.set_device(0)
train_dataset = RAPDataset("/home/dataset/human_attribute/RAP/RAP_annotation/train.txt",
                            transform = transforms.Compose([transforms.ToPILImage(),transforms.ColorJitter(0.05,0.05,0.05),transforms.RandomHorizontalFlip(),transforms.Resize(320),transforms.RandomCrop((299,299)),transforms.ToTensor()]))
test_dataset = RAPDataset("/home/dataset/human_attribute/RAP/RAP_annotation/test.txt",
                             transform = transforms.Compose([transforms.ToPILImage(),transforms.Resize(320),transforms.RandomCrop((299,299)),transforms.ToTensor()]))
train_loader = DataLoader(train_dataset,train_batch_size,True) #数据加载器  主要实现了批量加载 数据打乱 以及并行化读取数据 pytorch自带  也可以自己实现
test_loader = DataLoader(test_dataset,test_batch_size,True) 

#inception_v3 = Inception3(26,False,False) #实例化类LeNet  实际上也调用了LeNet的__init__()函数
hydraplus_net = HydraPlusNet(num_class,is_fusion = False)
#alex_model.initialize_parameters()
#从pytorch官方的预训练模型复制参数
inception_v3_pretrain = models.inception_v3(True)  #使用pytorch的预训练模型
pretrain_dict = inception_v3_pretrain.state_dict()
hydraplus_net_dict = hydraplus_net.state_dict()

pretrain_dict = {k:v for k,v in pretrain_dict.items() if k in hydraplus_net_dict}
hydraplus_net_dict.update(pretrain_dict)
hydraplus_net.load_state_dict(hydraplus_net_dict)


hydraplus_net = hydraplus_net.cuda()  #转移到GPU上
logger.info(hydraplus_net)

optimizer = torch.optim.SGD(hydraplus_net.parameters(),learning_rate,0.99,0.0005) #实例化一个SGD优化器 参数：需要更新的参数  学习率 momentum(默认为0) 正则化项系数(默认为0) dampening(动量抑制因子 默认为0) nesterov(是否使用Nesterov 默认为false)
#loss_func = torch.nn.CrossEntropyLoss()  #softmax loss函数
weight = torch.Tensor([1.7226262226969686, 2.6802565029531618, 1.0682133644154836, 2.580801475214588, 
1.8984257687918218, 2.046590013290684, 1.9017984669155032, 2.6014006200502586, 
2.272458988404639, 2.2625669787021203, 2.245380512162444, 2.3452980639899033, 
2.692210221689372, 1.5128949487853383, 1.7967419553099035, 2.5832221110933764, 
2.3302195718894034, 2.438480257574324, 2.6012705532709526, 2.704589108443237, 
2.6704246374231753, 2.6426970354162505, 1.3377813061118478, 2.284449325734624, 
2.417810793601295, 2.7015143874115033])
loss_func = torch.nn.BCEWithLogitsLoss()
loss_func.cuda()
start_epoch = 0
#读取断点训练参数
if use_save_parameter == True:
    checkpoint = torch.load(load_path)
    pretrain_dict = checkpoint["net"]
    model_dict = hydraplus_net.state_dict()
    pretrain_dict = {k:v for k,v in pretrain_dict.items() if k in model_dict}
    model_dict.update(pretrain_dict)
    hydraplus_net.load_state_dict(model_dict)
   # alex_model.load_state_dict(checkpoint["net"])
    if checkpoint_train == True:
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch =  checkpoint["epoch"] +1
    logger.info("success to load parameter")
train_loss = 0.0
train_acc = 0.0
test_loss = 0.0
test_acc = 0.0
num_iter = 0
recoder = {"num_iter":[],"train_loss":[],"train_acc":[],"test_loss":[],"test_acc":[]}

logger.info("start_epoch:%d",start_epoch)
for epoch in range(start_epoch,num_epoch):
    for batch_img,batch_label in train_loader:
        num_iter += 1
      #  print type(batch_img)
      #  print type(batch_label)

      #  print batch_label.shape
        batch_img = Variable(batch_img.cuda())   #将数据拷贝到GPU上运行，并且放入到Variable中
        batch_label = Variable(batch_label.cuda())
        hydraplus_net.train()
        output = hydraplus_net(batch_img) #进行前向计算
      #  print output  #output输出为batch_size*10 
      #  print batch_label.data.size()
        loss = loss_func(output,batch_label) #计算loss 此loss已经在一个batch_size上做了平均
      #  print loss.shape  #loss输出为一个标量
      #  train_loss += loss.item()  
        train_loss += loss.item()
        result = (output>= 0)  #输出大于等于0的设为1，小于0的设为0
        result = result.float()
       # print result
        train_correct = torch.sum((result == batch_label),0)  #统计一个batch中预测正确的数量
       # print "train_correct:",train_correct.size()
       # print train_acc
        train_correct = train_correct.cpu().numpy()  #转成numpy
        train_acc += train_correct  
      #  print "train_acc:",train_acc
        #反向传播
        hydraplus_net.zero_grad()  #因为梯度是累计的 所以先将梯度清0
        loss.backward()  #将自动计算所有可学习参数的梯度
        optimizer.step()  #调用优化器来更新参数


        #显示设置
        if(num_iter % iter_display == 0 or num_iter == 1):
            logger.info("iter_num:%d",num_iter)
            if num_iter == 1:
                k = 1
            else:
                k = iter_display
            train_loss = train_loss/k
            train_acc = train_acc/float(k*train_batch_size*image_rate)
            recoder["num_iter"].append(num_iter)
            recoder["train_loss"].append(train_loss)
            recoder["train_acc"].append(train_acc)
            logger.info("train loss:%f",train_loss)
            logger.info("train acc:")
            for i in range(num_class):
                logger.info("%s:%f",label_name[i],train_acc[i])
            logger.info("mAP:%f",np.sum(train_acc)/num_class)
            train_loss = 0
            train_acc = 0
        
        #测试
        if(is_test and (num_iter % test_iterval == 0)):
            test_loss = 0
            test_acc = 0
            test_num_iter = 0
            hydraplus_net.eval()

            ##################
        #    TP = [0.0] *26
        #    P = [0.0] *26
        #    TN = [0.0] *26
        #    N = [0.0] *26
        #    Acc = 0.0
        #    Prec = 0.0
        #    Rec = 0.0
            ###################
            for test_batch_img,test_batch_label in test_loader:
            #####################
         #       Yandf = 0.1
         #       Yorf = 0.1
         #       Y = 0.1
         #       f = 0.1
            ######################
            #    print test_batch_img.shape
                test_num_iter  += 1
                #test_batch_img = test_batch_img.view(-1,test_batch_img.shape[2],test_batch_img.shape[3],test_batch_img.shape[4])
            #   test_batch_img = test_batch_img.view(-1,test_batch_img.shape[2],test_batch_img.shape[3],test_batch_img.shape[4])
            #    test_batch_label = test_batch_label.view(test_batch_label.shape[0]*test_batch_label.shape[1])
                #test_batch_label = test_batch_label.view(test_batch_label.shape[0])
                test_batch_img = Variable(test_batch_img.cuda())
                test_batch_label = Variable(test_batch_label.cuda())
              #  bs,ncrops,c,h,w = test_batch_img.size()
                output = hydraplus_net(test_batch_img)  #测试前向计算
              #  output_avg = output.view(bs,ncrops,-1).mean(1)
                loss = loss_func(output,test_batch_label)
                test_loss += loss.item()
               # test_loss += loss.data[0]
              #  print test_batch_label.data
                result = (output>=0)
                result = result.float()
                test_correct = torch.sum((result == test_batch_label),0)
                test_correct = test_correct.cpu().numpy()
                test_acc += test_correct
                #############################
                i = 0
                #print output.shape
               # for item in output[0]:
               #     if item.data[0] > 0:
               #         f = f+1
               #         Yorf = Yorf + 1
               #         if test_batch_label[0][i].data[0] == 1:
               #             TP[i] = TP[i] + 1
               #             P[i] = P[i] + 1
               #             Y = Y +1
               #             Yandf = Yandf + 1
               #         else:
               #             N[i] = N[i] + 1
               #     else:
               #         if test_batch_label[0][i].data[0] == 0:
               #             TN[i] = TN[i] + 1
               #             N[i] = N[i] + 1
               #         else:
               #             P[i] = P[i] + 1
               #             Yorf = Yorf + 1
               #             Y = Y +1
               #     i = i +1
               # Acc = Acc + Yandf/Yorf
               # Prec = Prec + Yandf/f
               # Rec = Rec + Yandf/Y
                #################################
            test_loss = test_loss/test_num_iter
            test_acc = test_acc/float(len(test_dataset))
            recoder["test_loss"].append(test_loss)
            recoder["test_acc"].append(test_acc)
            logger.info("test loss:%f",test_loss)
            logger.info("test acc:")
            mAP = 0
            for i in range(num_class):
                logger.info("%s:%f",label_name[i],test_acc[i])
            logger.info("mAP:%f",np.sum(test_acc)/num_class)
            #####################################
        #    Accuracy = 0
        #    for l in range(26):
        #    #    print("%s : %f" %(classes[l],(TP[l]/P[l] + TN[l]/N[l])/2))
        #        Accuracy = TP[l]/P[l] + TN[l]/N[l] + Accuracy
        #    meanAccuracy = Accuracy/52
            
        #    Acc = Acc/10000
        #    Prec = Prec/10000
        #    Rec = Rec/10000
        #    F1 = 2*Prec*Rec/(Prec+Rec)
        #    print "mA:",meanAccuracy
        #    print "Acc:",Acc
        #    print "Prec:",Prec
        #    print "Rec:",Rec
        #    print "F1:",F1

        #参数保存
        if(num_iter % iter_save == 0):
            state = {"net":hydraplus_net.state_dict(),"optimizer":optimizer.state_dict(),"epoch":epoch}
            full_save_path = save_path +"_" + str(num_iter) +".weight"
            torch.save(state,full_save_path)
            logger.info("success to save parameter")
#训练完成 再进行一次存储权重
state = {"net":hydraplus_net.state_dict(),"optimizer":optimizer.state_dict(),"epoch":epoch}
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
