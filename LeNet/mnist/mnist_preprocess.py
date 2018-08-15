#!/usr/bin/env python
# coding=utf-8

import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument("--mnist_dataset_dir",help="the dataset of mnist",default="/home/dataset/mnist/images/")
#parser.add_argument("--label_name_txt",help="the name of cifar10 label",default="./label_name.txt")
args = parser.parse_args()

#1
#label_name_dic = {}
#f = open(args.label_name_txt,"r")
#for line in f:
#    strings = line.split(" ")
#    strings[1] = strings[1][:-1]
#    print strings[1]
#    label_name_dic[strings[0]] = strings[1]
#f.close()
#print label_name_dic

#2 train
train_lines = []
train_dir = os.path.join(args.mnist_dataset_dir,"train")
dirlist1 = os.listdir(train_dir)
for dir1 in dirlist1:
    path = os.path.join(train_dir,dir1)
    dirlist2 = os.listdir(path)
    for dir2 in dirlist2:
        img_dir = train_dir + "/" + dir1 + "/" + dir2
        line = img_dir + " " + dir1 +"\n"
        train_lines.append(line)

train_fw = open("./train.txt","w")
train_fw.writelines(train_lines)
train_fw.close()

#3 test
test_lines = []
test_dir = os.path.join(args.mnist_dataset_dir,"test")
dirlist1 = os.listdir(test_dir)
for dir1 in dirlist1:
    path = os.path.join(test_dir,dir1)
    dirlist2 = os.listdir(path)
    for dir2 in dirlist2:
        img_dir = test_dir + "/" + dir1 + "/" + dir2
        line = img_dir + " " + dir1 + "\n"
        test_lines.append(line)

test_fw = open("./test.txt","w")
test_fw.writelines(test_lines)
test_fw.close()

print "success to write file"
