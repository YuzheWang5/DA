import numpy as np
import p

# 假设 S 是你的输入矩阵，维度为 (36, N)
dataset = pd.read_csv("Q.csv", header=None)
S = ...

# N 是你的常数
N = 5

# 计算每一行的元素和并除以 N
row_sums = np.sum(S, axis=1)
q = row_sums[:, np.newaxis] / N

# 打印结果
print(q)