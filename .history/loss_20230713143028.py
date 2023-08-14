import numpy as np
import torch

# 读取CSV文件并转换为NumPy数组
S = np.loadtxt('Q.csv', delimiter=',')

# 将NumPy数组转换为PyTorch张量
tensor_data = torch.from_numpy(S)
print(S)
N = 5
row_sums = np.sum(S, axis=1)
q = row_sums[:, np.newaxis] / N

# 打印结果
print(q)
