import numpy as np

# 假设 S 是你的输入矩阵，维度为 (36, N)
S = ...

# N 是你的常数
N = ...

# 计算每一行的元素和并除以 N
row_sums = np.sum(S, axis=1)
q = row_sums[:, np.newaxis] / N

# 打印结果
print(q)