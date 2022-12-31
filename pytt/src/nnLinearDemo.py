import torch
import torch.nn as nn


# 首先初始化一个全连接神经网络
full_connected = nn.Linear(12, 15)

# 输入
input = torch.randn(5, 12)

# 输出
output = full_connected(input)

print(output.shape)
print(full_connected.weight.shape)
print(full_connected.bias.shape)

# 结果展示 nn.Linear往往用来初始化矩阵，供神经网络使用
# output.shape:torc h.Size([5, 15])
# full_connected.weight.shape: torch.Size([15, 12])
# full_connected.bias.shape: torch.Size([15])