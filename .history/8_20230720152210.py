import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Generated Q martix randomly
Q = np.zeros(12)
segment_sum = 2
for i in range(1):
    valid = False
    while not valid:
        temp = np.abs(np.random.rand(12))
        temp = temp * segment_sum / np.sum(temp)
        if np.abs(np.sum(temp * 2 / np.sum(temp)) - np.sum(temp)) < 1e-6:
            valid = True

    Q[i*12:(i+1)*12] = temp

#define input
dataset = Q
#target_dataset = pd.read_csv("Q_target.csv", header=None)

input_tensor = torch.tensor(dataset, dtype=torch.float32, requires_grad=True)
print(input_tensor.shape)