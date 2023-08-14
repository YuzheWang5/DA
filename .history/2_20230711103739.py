import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

dataset = pd.read_csv('Q.csv', header=None)
input_tensor = torch.tensor(dataset.values, dtype=torch.float32)

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(12, 256, batch_first=True)
        self.fc = nn.Linear(256, 12)

    def forward(self, x):
        output, _ = self.rnn(x)
        output = self.fc(output[:, -1, :])
        return output

rnn_model = RNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(rnn_model.parameters(), lr=0.001)

output_list = []
for _ in range(1000):
    output = rnn_model(input_tensor.unsqueeze(0))
    output_array = output.detach().numpy()
    output_list.append(output_array)

    #loss = criterion(output, input_tensor.unsqueeze(0))
    #optimizer.zero_grad()
    loss.backward()
    optimizer.step()

output = rnn_model(input_tensor.unsqueeze(0))
output_array = output.detach().numpy()
output_df = pd.DataFrame(output_array)
output_df.to_csv('output3.csv', header=None, index=None)