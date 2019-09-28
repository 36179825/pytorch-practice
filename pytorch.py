# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 10:03:12 2019

@author: Jasmine
"""

import torch
import numpy as np
from torch.autograd import Variable

"""
np.arrange(起始, 終止, 間隔)
e.g. 
np.arange(1,3,0.3)
array([ 1. ,  1.3,  1.6,  1.9,  2.2,  2.5,  2.8])
"""
np_data = np.arange(6).reshape((2, 3))
torch_data = torch.from_numpy(np_data)
tensor2array = torch_data.numpy()
print(
    '\nnumpy array:', np_data,          # [[0 1 2], [3 4 5]]
    '\ntorch tensor:', torch_data,      #  0  1  2 \n 3  4  5    [torch.LongTensor of size 2x3]
    '\ntensor to array:', tensor2array, # [[0 1 2], [3 4 5]]
)

# abs, 絕對值
data = [-1, -2, 1, 2]
tensor = torch.FloatTensor(data)  # 32-bit floating point
print(
    '\nabs',
    '\nnumpy: ', np.abs(data),          # [1 2 1 2]
    '\ntorch: ', torch.abs(tensor)      # [1 2 1 2]
)

# sin, 三角函數sin
print(
    '\nsin',
    '\nnumpy: ', np.sin(data),      # [-0.84147098 -0.90929743  0.84147098  0.90929743]
    '\ntorch: ', torch.sin(tensor)  # [-0.8415 -0.9093  0.8415  0.9093]
)

# mean, 平均
print(
    '\nmean',
    '\nnumpy: ', np.mean(data),         # 0.0
    '\ntorch: ', torch.mean(tensor)     # 0.0
)

# matrix multiplication
data = [[1,2], [3,4]]
tensor = torch.FloatTensor(data)  # 32-bit floating point
# correct method
print(
    '\nmatrix multiplication (matmul)',
    '\nnumpy: ', np.matmul(data, data),     # [[7, 10], [15, 22]]
    '\ntorch: ', torch.mm(tensor, tensor)   # [[7, 10], [15, 22]]
)
# incorrect method
data = np.array(data)
print(
    '\nmatrix multiplication (dot)',
    '\nnumpy: ', data.dot(data),        # [[7, 10], [15, 22]]
    '\ntorch: ', tensor.dot(tensor)     # this will convert tensor to [1,2,3,4], you'll get 30.0
)

# Variable in torch is to build a computational graph,
# but this graph is dynamic compared with a static graph in Tensorflow or Theano.
# So torch does not have placeholder, torch can just pass variable to the computational graph.

# 先生鸡蛋
tensor = torch.FloatTensor([[1,2],[3,4]])
# 把鸡蛋放到篮子里, requires_grad是参不参与误差反向传播, 要不要计算梯度
variable = Variable(tensor, requires_grad=True)

print(tensor)       # [torch.FloatTensor of size 2x2]
print(variable)     # [torch.FloatTensor of size 2x2]

# till now the tensor and variable seem the same.
# However, the variable is a part of the graph, it's a part of the auto-gradient.

t_out = torch.mean(tensor*tensor)       # x^2
v_out = torch.mean(variable*variable)   # x^2
print(t_out)
print(v_out)    # 7.5

"""
v_out = torch.mean(variable*variable) 就是在计算图中添加的一个计算步骤, 计算误差反向传递的时候有他一份功劳。
下面两步看不懂没关系, 只要知道 Variable 是计算图的一部分, 可以用来传递误差就好.
v_out = 1/4 * sum(variable*variable) 这是计算图中的 v_out 计算步骤
针对于 v_out 的梯度就是, d(v_out)/d(variable) = 1/4*2*variable = variable/2

"""

v_out.backward()    # backpropagation from v_out
# v_out = 1/4 * sum(variable*variable)
# the gradients w.r.t the variable, d(v_out)/d(variable) = 1/4*2*variable = variable/2
print(variable.grad)
'''
 0.5000  1.0000
 1.5000  2.0000
'''

"""
直接print(variable)只会输出 Variable 形式的数据, 在很多时候是用不了的(比如想要用 plt 画图), 
所以我们要转换一下, 将它变成 tensor 形式.
"""

print(variable)     # this is data in variable format
"""
Variable containing:
 1  2
 3  4
[torch.FloatTensor of size 2x2]
"""

print(variable.data)    # this is data in tensor format
"""
 1  2
 3  4
[torch.FloatTensor of size 2x2]
"""

print(variable.data.numpy())    # numpy format
"""
[[ 1.  2.]
 [ 3.  4.]]
"""