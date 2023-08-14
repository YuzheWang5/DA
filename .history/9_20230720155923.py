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

#define input
dataset = Q

input_tensor = torch.tensor(dataset, dtype=torch.float32, requires_grad=True).unsqueeze(0)

# define RNN
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        q0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        output, _ = self.rnn(x, q0)
        output = self.fc(output[:, -1, :])
        output = torch.sigmoid(output)
        return output.unsqueeze(-1)

sequence_length = 12
input_size = 1
hidden_size = 256
num_layers = 1
output_size = 12
num_epochs = 100
learning_rate = 0.01

class EulideanLoss(nn.Module):
    def __init__(self):
        super(EulideanLoss, self).__init__()

    def forward(self, Q, q):
        return torch.sqrt(torch.sum((Q-q)**2))

model = RNN(input_size, hidden_size, num_layers, output_size)
criterion = EulideanLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

all_Sn = []
losses = []
qn=

for epoch in range(num_epochs):
    model.train()
    outputs = model(input_tensor)
    all_outputs.append(outputs.detach())
    q = torch.mean(torch.stack(all_outputs),dim=0)
    loss = criterion(q, input_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

    print(f"Epoch: {epoch+1}/{num_epochs}, Loss: {loss.item()}")

plt.plot(losses)
plt.xlabel('Time')
plt.ylabel('Loss')
plt.show()

all_outputs_np = [output.numpy() for output in all_outputs]
all_outputs = np.concatenate(all_outputs, axis=0)
pd.DataFrame(all_outputs).to_csv('output.csv', index=False,header=False)