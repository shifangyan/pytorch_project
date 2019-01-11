#!/usr/bin/env python
# coding=utf-8

from torch.utils.data import DataLoader
from GlassesDataset import GlassesDataset
from VggNet import *
from torchvision import transforms
from torchvision import utils
import skimage.io as io
import numpy as np

use_dataset = True
dataset_root_path = "/dataset/human_attribute/"
dataset_class = GlassesDataset
test_label_txt = "/dataset/human_attribute/glassess/test.txt"
batch_size = 32
model_function = vggnet
weight_path = "./weight/AlexNet_500.weight" 
#weight_path = "../ResNet/weight/VggNet_13200.weight"
use_gpu = True
gpu_ids = [1,2,3,7]
label = ["not wearing glasses","wearing glasses"]
test_img_path = "../hydraplus_net_new/test_image/zyx.png"

class ModelTest(object):
    #def __init__(self):

    def HyperparameterInitialization(self):
        self.use_dataset = use_dataset
        self.dataset_root_path = dataset_root_path
        self.dataset_class = dataset_class
        self.test_label_txt = test_label_txt
        self.batch_size = batch_size
        self.use_gpu = use_gpu
        self.gpu_ids = gpu_ids
        if not isinstance(self.gpu_ids,list):
            print "error:gpu_idx参数类型错误"
            exit(0)
        self.label = label

    def DataInitialization(self,transform = None):
        if self.use_dataset:
            dataset = dataset_class(self.test_label_txt,self.dataset_root_path,transform)
            self.test_loader = DataLoader(dataset,self.batch_size,True)
        else:
            self.transform = transform

    def ModelInitialization(self,model_function,weight_path = None,last_liner_name = "classifier_liner",num_classes = 2,bn= True,model = "D"):
        if weight_path is None:
            print "please give weight_path"
            exit()
        self.model = model_function(True,weight_path,last_liner_name,num_classes,bn,model)
        if self.use_gpu:
            torch.cuda.set_device(self.gpu_ids[0])
            self.model = self.model.cuda()
            self.model = torch.nn.DataParallel(self.model,gpu_ids)
        print self.model
    def Test(self,img = None):
        if self.use_dataset:
            print "start test"
            self.test_acc = 0
            test_iter = 0
            self.model.eval()
            for batch_img,batch_label in self.test_loader:
                test_iter += 1 
                if self.use_gpu:
                    batch_img = batch_img.cuda()
                    batch_label = batch_label.cuda()
                output = self.model(batch_img)
                max,max_index = torch.max(output,1)
                test_currect = torch.sum((max_index == batch_label))
                self.test_acc += test_currect

            self.test_acc = float(self.test_acc)/float(test_iter * self.batch_size)
            print "test acc:",self.test_acc
        else:
            if img is None:
                print "please give image"
                exit(0)
            if self.transform is not None:
                img = self.transform(img)
                img = torch.unsqueeze(img,0)
            if self.use_gpu:
                img = img.cuda()
            output = self.model(img)
            max,max_index = torch.max(output,1)
            #print max_index.squeeze().size()
            print self.label[max_index.squeeze().item()]
            #print output
            utils.make_grid(img)
if __name__ == "__main__":
    model_test = ModelTest()
    model_test.HyperparameterInitialization()
    transform = transforms.Compose([transforms.ToPILImage(),transforms.Resize(256),transforms.CenterCrop(227),transforms.ToTensor()])
    model_test.DataInitialization(transform)
    model_test.ModelInitialization(vggnet,weight_path = weight_path,last_liner_name = "wearing_glasses",num_classes = 2,bn = True)
    img = io.imread(test_img_path)
    model_test.Test(img)
