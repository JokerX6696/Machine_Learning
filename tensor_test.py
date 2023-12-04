#!D:/Application/python/python.exe
import torch
import numpy as np
#  张量创建
arr = np.ones((3,3))
t = torch.tensor(arr,device='cuda')  # 将张量创建在 cuda 上
t = torch.from_numpy(arr)  #  from_numpy 无法在 GPU 上创建张量 是因为函数设计是共享内存

torch.zeros()

# 直接创建：

torch.tensor(data, dtype=None, device=None, requires_grad=False): #从给定的数据直接创建张量。
torch.zeros(size, dtype=None, device=None, requires_grad=False): #创建一个所有元素为0的张量。
torch.ones(size, dtype=None, device=None, requires_grad=False): #创建一个所有元素为1的张量。
torch.full(size, fill_value, dtype=None, device=None, requires_grad=False): #创建一个所有元素都为指定值的张量。
torch.eye(n, m=None, dtype=None, device=None, requires_grad=False): #创建一个单位矩阵。
#从 NumPy 数组创建：

torch.from_numpy(ndarray)#: 从 NumPy 数组创建张量，这个张量和原始的 NumPy 数组共享内存。
#特殊张量：

torch.arange(start=0, end, step=1, dtype=None, device=None, requires_grad=False)#: 创建一个等差数列的张量。
torch.linspace(start, end, steps=100, dtype=None, device=None, requires_grad=False)#: 创建一个在指定范围内均匀分布的张量。
torch.logspace(start, end, steps=100, base=10.0, dtype=None, device=None, requires_grad=False)#: 创建一个在指定范围内对数均匀分布的张量。
#随机张量：

torch.rand(size, dtype=None, device=None, requires_grad=False)#: 创建一个在[0, 1)范围内均匀分布的随机张量。
torch.randn(size, dtype=None, device=None, requires_grad=False)#: 创建一个从标准正态分布中抽取的随机张量。
torch.randint(low, high, size, dtype=None, device=None, requires_grad=False)#: 创建一个在指定范围内整数均匀分布的随机张量。
#使用特殊初始化方法：

torch.nn.init# 模块提供了各种权重初始化方法，例如 torch.nn.init.xavier_uniform_、torch.nn.init.normal_ 等。
# 张量拼接与切分
# 拼接
torch.cat()  # 不改变张量的维度
torch.stack() # 建立新的维度进行数据拼接
# 切分
torch.chunk() # 将张量按维度进行平均切分
torch.split() # 将张量按照维度进行切分

torch.tensor([1,2,3])
torch.index_select() # 在维度上，按照 index 索引数据
int_tensor = torch.randint(0,9,size=(3,3,3))
idx = torch.tensor([0,2], dtype = torch.int32)
new = torch.index_select(int_tensor,dim=1,index=idx)  # 索引 dim 为 几  就在第几个维度进行索引， 直观感受是按照第几个括号进行索引！！！
torch.mask(t,mask) # 返回满足条件的一位向量
# 改变形状
torch.reshape()  # 变化张量的形状