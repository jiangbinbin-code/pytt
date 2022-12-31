import torch, torchvision

model = torchvision.models.resnet18(pretrained=True)
data = torch.rand(1, 3, 64, 64)
labels = torch.rand(1, 1000)
# 张量
x = torch.rand(4)
# 二维张量
x = torch.rand(2, 4)

print(data)
print(labels)
# 模型的每一层运行输入数据以进行预测。 这是正向传播
prediction = model(data)  # forward pass
# 使用模型的预测和相应的标签来计算误差（loss）。
# 下一步是通过网络反向传播此误差。 当我们在误差张量上调用.backward()时，开始反向传播。
# 然后，Autograd 会为每个模型参数计算梯度并将其存储在参数的.grad属性中
loss = (prediction - labels).sum()
loss.backward()  # backward pass
# 加载一个优化器，在本例中为 SGD，学习率为 0.01，动量为 0.9。 我们在优化器中注册模型的所有参数
optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
# 调用.step()启动梯度下降
optim.step()  # gradient descent



import torch

a = torch.tensor([2., 3.], requires_grad=True)
b = torch.tensor([6., 4.], requires_grad=True)

Q = 3*a**3 - b**2


external_grad = torch.tensor([1., 1.])
Q.backward(gradient=external_grad)


# check if collected gradients are correct
print(9*a**2 == a.grad)
print(-2*b == b.grad)
