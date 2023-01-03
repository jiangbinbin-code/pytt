import torch
from torch import tensor

'''
 torch.no_grad :
 逃避自动梯度下降检测
 可以让节点不进行求梯度,从而节省了内存控件,当神经网络较大且内存不够用时,就需要让梯度为False
'''

a = torch.ones(2, requires_grad=True)
b = a * 2
print(a, a.grad, a.requires_grad)
b.sum().backward(retain_graph=True)
print(a, a.grad, a.requires_grad)
with torch.no_grad():
    a = a + a.grad
    print(a, a.grad, a.requires_grad)
    # a.grad.zero_()
b.sum().backward(retain_graph=True)
print(a, a.grad, a.requires_grad)

# tensor([1., 1.], requires_grad=True) None True
# tensor([1., 1.], requires_grad=True) tensor([2., 2.]) True
# tensor([3., 3.]) None False
# tensor([3., 3.]) None False
