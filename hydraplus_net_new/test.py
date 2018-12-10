#!/usr/bin/env python
# coding=utf-8

from torch.utils.data import DataLoader
import torch
from PA100KDataset import PA100KDataset
from HydraPlusNet import *
import numpy as np
from torchvision import transforms
from torch.autograd import Variable

test_batch_size = 1 
load_path = "./weight/InceptionV3_5000.weight"


label_name = ["Female","AgeOver60","Age18-60","AgeLess18","Front","Side",
              "Back","Hat","Glasses","HandBag","ShoulderBag","Backpack","HoldObjectsInFront",
              "ShortSleeve","LongSleeve","UpperStride","UpperLogo","UpperPlaid","UpperSplice",
              "LowerStripe","LowerPattern","LongCoat","Trousers","Shorts","Skirt&Dress","boots"]

torch.cuda.set_device(0)
test_dataset = PA100KDataset("/home/dataset/human_attribute/PA100K/test.txt",
                             transform = transforms.Compose([transforms.ToPILImage(),transforms.Resize(320),transforms.RandomCrop((299,299)),transforms.ToTensor()]))
test_loader = DataLoader(test_dataset,test_batch_size,True)

hydraplus_net = HydraPlusNet(26,is_fusion = True)

checkpoint = torch.load(load_path)
hydraplus_net.load_state_dict(checkpoint["net"])

test_loss = 0
test_acc = 0
test_num_iter = 0

hydraplus_net.cuda()
hydraplus_net.eval()
for test_batch_img,test_batch_label in test_loader:
    test_num_iter += 1
    test_batch_img = Variable(test_batch_img.cuda())
    test_batch_label = Variable(test_batch_label.cuda())
   # bs,ncrops,c,h,w = test_batch_img.size()
    output = hydraplus_net(test_batch_img)
   # output_avg = output.view(bs,ncrops,-1).mean(1)
    result = (output>=0)
    result = result.float()
    test_correct = torch.sum((result == test_batch_label),0)
    test_correct = test_correct.cpu().numpy()
    test_acc += test_correct
test_acc = test_acc/float(len(test_dataset))
print "test acc:"
mAP = 0
for i in range(26):
    print label_name[i],":",test_acc[i]
print "mAP:",(np.sum(test_acc)/26)
