# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 17:20:33 2018

@author: Administrator
"""

from urllib import request,error
import re
import os

#定义搜索词并将搜索词转码，防止报错
key_word = "脸"
key_name=request.quote(key_word)
if not os.path.exists(key_word):
    os.mkdir(key_word)
#定义函数，将爬到的每一页的商品url写入到文件
def savefile(data):
    path= os.path.join(key_word,"url.txt")
    file=open(path,"a")
    file.write(data+"\n")
    file.close()
#外层for循环控制爬取的页数 将每页的url写入到本地
for p in range(0,60):
#拿到每页url
    url="https://s.taobao.com/search?q=" + key_name + "&s=" + str(p*48)
#拿到每页源码
    data1=request.urlopen(url).read().decode("utf-8")
#调用函数savefile,将每页url存入到指定path
    savefile(url)
    #定义匹配规则
    pat='pic_url":"//(.*?)"'
    #匹配到的所有图片url
    img_url=re.compile(pat).findall(data1)
    print(img_url)
    #内层for循环将所有图片写到本地
    for a_i in range(0,len(img_url)):
        this_img=img_url[a_i]
        this_img_url="http://"+this_img
        #每张图片的url
        print(this_img_url)
        #将每张图片写到本地
        #定义存取本地图片路径【retrieve()不会再本地建立文件夹因此需要手建】
        img_path= key_word + "/" + str(p)+ str(a_i)+".jpg"
        request.urlretrieve(this_img_url,img_path)

#本文来自 KevinMahone 的CSDN 博客 ，全文地址请点击：https://blog.csdn.net/Aaron_Miller/article/details/80278627?utm_source=copy 
