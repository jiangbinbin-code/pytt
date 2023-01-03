import torch
import math

'''
线性间距向量
torch.linspace(start, end, steps=100, out=None) → Tensor
返回一个1维张量，包含在区间start和end上均匀间隔的step个点。
输出张量的长度由steps决定。
'''
x = torch.linspace(-math.pi, math.pi, 2000)
y = torch.sin(x)

p = torch.tensor([1, 2, 3])
xx = x.unsqueeze(-1).pow(p)
'''
  nn.Linear 全连接层
  torch.flatten(x)等于torch.flatten(x，0)默认将张量拉成一维的向量，也就是说从第一维开始平坦化，torch.flatten(x，1)代表从第二维开始平坦化。
  按照上边的说法，与一层一层的单独调用模块组成序列相比，nn.Sequential() 可以允许将整个容器视为单个模块（即相当于把多个模块封装成一个模块），
  forward()方法接收输入之后，nn.Sequential()按照内部模块的顺序自动依次计算并输出结果。
'''
model = torch.nn.Sequential(
    torch.nn.Linear(3, 1),
    torch.nn.Flatten(0, 1)
)
''' 求损失函数 '''
loss_fn = torch.nn.MSELoss(reduction='sum')

learning_rate = 1e-6
for t in range(2000):

    y_pred = model(xx)

    loss = loss_fn(y_pred, y)
    if t % 100 == 99:
        print(t, loss.item())
    model.zero_grad()
    loss.backward()
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad

linear_layer = model[0]

print(f'Result: y = {linear_layer.bias.item()} + {linear_layer.weight[:, 0].item()} x + {linear_layer.weight[:, 1].item()} x^2 + {linear_layer.weight[:, 2].item()} x^3')
