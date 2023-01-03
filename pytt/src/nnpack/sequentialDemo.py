from collections import OrderedDict

from torch import nn


class net(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(net, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(in_channel, in_channel / 4, kernel_size=1),
                                    nn.BatchNorm2d(in_channel / 4),
                                    nn.ReLU())
        self.layer2 = nn.Sequential(nn.Conv2d(in_channel / 4, in_channel / 4),
                                    nn.BatchNorm2d(in_channel / 4),
                                    nn.ReLU())
        self.layer3 = nn.Sequential(nn.Conv2d(in_channel / 4, out_channel, kernel_size=1),
                                    nn.BatchNorm2d(out_channel),
                                    nn.ReLU())

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x


# 一个序列容器，用于搭建神经网络的模块被按照被传入构造器的顺序添加到nn.Sequential()容器中。除此之外，
# 一个包含神经网络模块的OrderedDict也可以被传入nn.Sequential()容器中。利用nn.Sequential()搭建好模型架构，
# 模型前向传播时调用forward()方法，模型接收的输入首先被传入nn.Sequential()包含的第一个网络模块中。然后，
# 第一个网络模块的输出传入第二个网络模块作为输入，按照顺序依次计算并传播，直到nn.Sequential()里的最后一个模块输出结果。


model = nn.Sequential(
          nn.Conv2d(1,20,5),
          nn.ReLU(),
          nn.Conv2d(20,64,5),
          nn.ReLU()
        )

# Using Sequential with OrderedDict. This is functionally the
# same as the above code
model = nn.Sequential(OrderedDict([
          ('conv1', nn.Conv2d(1,20,5)),
          ('relu1', nn.ReLU()),
          ('conv2', nn.Conv2d(20,64,5)),
          ('relu2', nn.ReLU())
        ]))
