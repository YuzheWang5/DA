import numpy as np
import torch

# 读取CSV文件并转换为NumPy数组
S = np.loadtxt('Q.csv', delimiter=',')

# 将NumPy数组转换为PyTorch张量
tensor_data = torch.from_numpy(S)
