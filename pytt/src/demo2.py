import torch
import numpy as np

x = torch.randn(3, 1, 5, 4)
# print(x)

conv = torch.nn.Conv2d(1, 4, (2, 3))
res = conv(x)

# print(res.shape)  # torch.Size([3, 4, 4, 2])




data = np.arange(12).reshape(2, 6)  # 生成数据并重组成2行六列
print(data)
'''
[[ 0  1  2  3  4  5]
 [ 6  7  8  9 10 11]]
'''
cols = data.shape[0]  # 0表示行数
print(cols)  # 2
cols1 = data.shape[1]  # 1表示列数
print(cols1)  # 6

shape = (2, 3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")
