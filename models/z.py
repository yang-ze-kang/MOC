import pickle
import numpy as np
import cv2
from utils.colors import colormap
import torch

import torch.optim.adam


a = torch.tensor([[1,2,3]])
b = torch.tensor([[2,1,2]])

x = torch.tensor([[0,0],[4,5],[7,8]],dtype=float)
y = torch.tensor([[11,2],[41,5],[71,8]])

# deta = x-y
# print(torch.mean(x,dim=1).unsqueeze(1))
# x = (x-torch.mean(x,dim=1).unsqueeze(1))/(torch.std(x,dim=1).unsqueeze(1)+1e-7)
# print(x)

print(x.shape)
x = x/torch.sqrt(torch.sum(x*x,dim=1)).unsqueeze(1)
print(torch.sqrt(torch.sum(x*x,dim=1)).unsqueeze(1))


