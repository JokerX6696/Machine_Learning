#!D:/Application/python/python.exe
import torch
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(1311)

lr = 0.01  # 学习率

# 创建训练数据   初始化随机值
x = torch.rand(30,1) * 10
y = 2*x + (5 + torch.randn(30,1))

# 构建线性回归参数    初始化随机值
w = torch.randn((1),requires_grad=True)
b = torch.randn((1),requires_grad=True)


for iteration in range(0,1000):

    # 前向传播
    wx = torch.mul(w,x)  # 相乘
    y_pred = torch.add(wx,b)  # 相加得到 y 预测值
    # 计算损失值
    loss = (0.5 * (y - y_pred) ** 2).mean()

    # 反向传播
    loss.backward()

    # 更新参数
    b.data.sub_(lr * b.grad)
    w.data.sub_(lr * w.grad)

    if loss.data.numpy() < 1:
        break

plt.scatter(x.data.numpy(),y.data.numpy())
plt.plot(x.data.numpy(),y_pred.data.numpy())
plt.text(2,20,'Loss=%.4f' %loss.data.numpy(), fontdict= {'size':20,'color':'red'} )
plt.xlim(1.5,10)
plt.ylim(8,28)
plt.title("Iteration: {}\nw : {}  b : {}".format(iteration,w.data.numpy(),b.data.numpy()))
plt.pause(0.5)
plt.savefig('D:/desk/github/Machine_Learning/my_plot.png')