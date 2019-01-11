#!/usr/bin/env python
# coding=utf-8

from torch.utils.data import DataLoader
import torch
#from DogsDataset import DogsDataset
from GlassesDataset import GlassesDataset
from InceptionV3 import * 
import skimage.io as io
from torchvision import transforms
from torchvision import models
from torch import nn
import logging
import time

learning_rate = 0.001
momentum = 0.9
regularization_parameter = 0.0005
use_gpu = True
gpu_ids = [1,2,3,7]
train_batch_size = 64  #60000/64=937.5
validate_batch_size = 16
image_rate = 1  #每张图片经过预处理扩展的倍数
iter_display = 100 #迭代多少次显示一次
num_epoch = 250
is_validate = True
validate_iterval = 500  #每训练多少次进行一次测试
iter_save = 500 #美巡多少次进行一次参数保存  可用于断点训练
save_path = "./weight/InceptionV3"   #参数保存路径 会在后面加后缀 例如 ./weight/AlexNet_1000.weight
load_path = "./weight/AlexNet_8000.weight"  #参数读取路径
use_save_parameter = False  #决定是否使用保存参数
checkpoint_train = False  #决定是否进行断点训练
train_label_txt = "/dataset/human_attribute/glassess/train.txt"
validate_label_txt = "/dataset/human_attribute/glassess/test.txt"
dataset_root_path = "/dataset/human_attribute/"
dataset_class = GlassesDataset
model_function = inception_v3 
#test_iter = 150  #测试进行多少次 10000/64 = 156 

class ModelTrain(object):
    def __init__(self):
        self.LoggerInitialization()

    def HyperparameterInitialization(self):
        self.initial_learning_rate = learning_rate
        self.momentum = momentum
        self.regularization_parameter = regularization_parameter
        self.use_gpu = use_gpu
        self.gpu_ids = gpu_ids
        if not isinstance(self.gpu_ids,list):
            self.logger.info("error:gpu_ids参数类型错误")
            exit()
        self.train_batch_size = train_batch_size
        self.validate_batch_size = validate_batch_size
        self.iter_display = iter_display
        self.num_epoch = num_epoch
        self.is_validate = is_validate
        self.validate_iterval = validate_iterval
        self.iter_save = iter_save
        self.save_path = save_path
        self.load_path = load_path
        self.train_label_txt = train_label_txt
        self.validate_label_txt = validate_label_txt
        self.dataset_root_path = dataset_root_path
        self.dateset_class = dataset_class
        self.iter = 0
        self.train_loss = 0.0
        self.train_acc = 0.0
        self.validate_loss = 0.0
        self.validate_acc = 0.0
    def DataInitialization(self,train_transform = None,validate_transform = None):
        train_dataset = dataset_class(self.train_label_txt,self.dataset_root_path,train_transform)
        validate_dataset = dataset_class(self.validate_label_txt,self.dataset_root_path,validate_transform)
        self.train_loader = DataLoader(train_dataset,self.train_batch_size,True)
        self.validate_loader = DataLoader(validate_dataset,self.validate_batch_size,True)
    def ModelInitialization(self,model_function,fine_tuning,weight_path = None,last_liner_name = "classifier_liner",
                           num_classes = 2):
        self.model = model_function(fine_tuning,weight_path,last_liner_name,num_classes)
        if self.use_gpu: #使用GPU设备
            torch.cuda.set_device(self.gpu_ids[0])
            self.model = self.model.cuda()
            self.model = torch.nn.DataParallel(self.model,gpu_ids)
        self.logger.info(self.model)

    def OptimizerInitialization(self):
        self.optimizer = torch.optim.SGD(self.model.parameters(),self.initial_learning_rate,self.momentum,self.regularization_parameter)
        self.loss_func = torch.nn.CrossEntropyLoss()
    def LoggerInitialization(self):
        #日志设置
        #第一步 创建一个logger
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)

        #第二步 创建两个handler 一个写入日志文件 一个向屏幕打印
        rq = time.strftime("%Y%m%d%H%M",time.localtime(time.time()))
        log_name = "./" + rq + ".log"
        file_handler = logging.FileHandler(log_name,mode = "w")
        file_handler.setLevel(logging.INFO)

        print_handler = logging.StreamHandler()
        print_handler.setLevel(logging.DEBUG)

        #第三步 设置两个handler的输出格式
        file_formatter = logging.Formatter("%(asctime)s-%(filename)s[line:%(lineno)d] - %(levelname)s:%(message)s")
        file_handler.setFormatter(file_formatter)

        #第四步 将handler添加到Logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(print_handler)

    def MainLoop(self):
        self.logger.info("start train")
        for epoch in range(self.num_epoch):
            for batch_img,batch_label in self.train_loader:

                self.TrainStep(batch_img,batch_label)
                #显示设置
                if(self.iter % self.iter_display == 0 or self.iter == 1):
                    self.DisplayStep()
                if(self.iter % self.validate_iterval == 0):
                    self.ValidateStep()
                if(self.iter % self.iter_save == 0):
                    self.ModelSaveStep()
        self.ModelSaveStep()
    def TrainStep(self,batch_img,batch_label):
        self.iter += 1
        self.model.train()
        if self.use_gpu:
            batch_img = batch_img.cuda()
            batch_label = batch_label.cuda()
        output = self.model(batch_img)
        loss = self.loss_func(output,batch_label)
        self.train_loss += loss.item()
        max,max_index = torch.max(output,1)
        train_correct = torch.sum((max_index == batch_label))
        self.train_acc += train_correct

        #反向传播
        self.model.zero_grad()  #因为梯度是累计的 所以先将梯度清0
        loss.backward()       #自动计算所有课学习参数的梯度
        self.optimizer.step()  #调用优化器来更新参数

    def ValidateStep(self):
        self.validate_loss = 0
        self.validate_acc = 0
        validate_iter = 0
        self.model.eval()
        for validate_batch_img,validate_batch_label in self.validate_loader:
            validate_iter += 1
            if self.use_gpu:
                validate_batch_img = validate_batch_img.cuda()
                validate_batch_label = validate_batch_label.cuda()
            output = self.model(validate_batch_img)
            loss = self.loss_func(output,validate_batch_label)
            self.validate_loss += loss.item()
            max,max_index = torch.max(output,1)
            validate_currect = torch.sum((max_index == validate_batch_label))
            self.validate_acc += validate_currect

        self.validate_loss = self.validate_loss/validate_iter
        self.validate_acc = float(self.validate_acc)/float(validate_iter*self.validate_batch_size)
        self.logger.info("validate loss:%f",self.validate_loss)
        self.logger.info("validate acc:%f",self.validate_acc)
    def DisplayStep(self):
        self.logger.info("iter:%d",self.iter) 
        if self.iter == 1:
            iter_num = 1
        else:
            iter_num = self.iter_display
        self.train_loss = self.train_loss/iter_num
        self.train_acc = float(self.train_acc)/float(iter_num*self.train_batch_size)
        self.logger.info("train loss:%f",self.train_loss)
        self.logger.info("train acc:%f",self.train_acc)
    def ModelSaveStep(self):
        state = {"net":self.model.state_dict(),"optimizer":self.optimizer.state_dict()}
        weight_full_path = self.save_path + "_" + str(self.iter) + ".weight"
        torch.save(state,weight_full_path)
        self.logger.info("success to save weight")

if __name__ == "__main__":
    model_train = ModelTrain()
    model_train.HyperparameterInitialization()
    train_transform = transforms.Compose([transforms.ToPILImage(),transforms.Resize(256),transforms.RandomCrop(227),transforms.ToTensor()])
    validate_transform = transforms.Compose([transforms.ToPILImage(),transforms.Resize(256),transforms.CenterCrop(227),transforms.ToTensor()])
    model_train.DataInitialization(train_transform =train_transform,validate_transform = validate_transform)
    model_train.ModelInitialization(model_function,fine_tuning=False,weight_path = None,last_liner_name = "wearing_glasses",num_classes = 2)
    model_train.OptimizerInitialization()
#    model_train.LoggerInitialization()
    model_train.MainLoop()
