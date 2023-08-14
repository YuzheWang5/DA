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

# define RNN
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

