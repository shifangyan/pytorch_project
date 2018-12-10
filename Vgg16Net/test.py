#!/usr/bin/env python
# coding=utf-8

import torch
import torch.nn as nn
from collections import OrderedDict

a = OrderedDict()
a["conv1-1"] = nn.Conv2d(1,20,5)
a["relu1-1"] = nn.ReLU()
model = nn.Sequential(a)
#model = nn.Sequential(OrderedDict([("conv1",nn.Conv2d(1,20,5)),("relu1",nn.ReLU()),
#                                  ("conv2",nn.Conv2d(20,64,5)),("relu2",nn.ReLU())]))

print model

