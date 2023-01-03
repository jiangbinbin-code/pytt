import torch

'''
torch.flatten(x)等于torch.flatten(x，0)默认将张量拉成一维的向量，也就是说从第一维开始平坦化，torch.flatten(x，1)代表从第二维开始平坦化。
平坦化 就是多维一维化
'''
x = torch.randn(2, 4, 2)
print(x)
z = torch.flatten(x)
print(z)
w = torch.flatten(x, 1)
print(w)

'''
输出为：
tensor([[[-0.9814,  0.8251],
         [ 0.8197, -1.0426],
         [-0.8185, -1.3367],
         [-0.6293,  0.6714]],
 
        [[-0.5973, -0.0944],
         [ 0.3720,  0.0672],
         [ 0.2681,  1.8025],
         [-0.0606,  0.4855]]])
 
tensor([-0.9814,  0.8251,  0.8197, -1.0426, -0.8185, -1.3367, -0.6293,  0.6714,
        -0.5973, -0.0944,  0.3720,  0.0672,  0.2681,  1.8025, -0.0606,  0.4855])
 
tensor([[-0.9814,  0.8251,  0.8197, -1.0426, -0.8185, -1.3367, -0.6293,  0.6714]
,
        [-0.5973, -0.0944,  0.3720,  0.0672,  0.2681,  1.8025, -0.0606,  0.4855]
])

'''


'''
torch.flatten(x,0,1)代表在第一维和第二维之间平坦化
'''
x = torch.randn(2, 4, 2)
print(x)
w = torch.flatten(x, 0, 1)  # 第一维长度2，第二维长度为4，平坦化后长度为2*4
print(w.shape)
print(w)
'''
输出为：
tensor([[[-0.5523, -0.1132],
         [-2.2659, -0.0316],
         [ 0.1372, -0.8486],
         [-0.3593, -0.2622]],
 
        [[-0.9130,  1.0038],
         [-0.3996,  0.4934],
         [ 1.7269,  0.8215],
         [ 0.1207, -0.9590]]])
 
torch.Size([8, 2])
 
tensor([[-0.5523, -0.1132],
        [-2.2659, -0.0316],
        [ 0.1372, -0.8486],
        [-0.3593, -0.2622],
        [-0.9130,  1.0038],
        [-0.3996,  0.4934],
        [ 1.7269,  0.8215],
        [ 0.1207, -0.9590]])
'''

'''
对于torch.nn.Flatten()，因为其被用在神经网络中，输入为一批数据，第一维为batch，通常要把一个数据拉成一维，而不是将一批数据拉为一维。所以torch.nn.Flatten()默认从第二维开始平坦化
'''

# 随机32个通道为1的5*5的图
x = torch.randn(32, 1, 5, 5)

model = torch.nn.Sequential(
    # 输入通道为1，输出通道为6，3*3的卷积核，步长为1，padding=1
    torch.nn.Conv2d(1, 6, 3, 1, 1),
    torch.nn.Flatten()
)
output = model(x)
print(output.shape)  # 6*（7-3+1）*（7-3+1）

'''
 torch.Size([32, 150])
'''