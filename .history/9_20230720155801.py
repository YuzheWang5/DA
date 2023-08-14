import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Generated Q martix randomly
Q = np.zeros((12,1))
segment_sum = 2
for i in range(1):
    valid = False
    while not valid:
        temp = np.abs(np.random.rand(12,1))
        temp = temp * segment_sum / np.sum(temp)
        if np.abs(np.sum(temp * 2 / np.sum(temp)) - np.sum(temp)) < 1e-6:
            valid = True
Q[i*12:(i+1)*12] = temp

