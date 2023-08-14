import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

dataset = pd.read_csv("Q.csv", header=None)
#target_dataset = pd.read_csv("Q_target.csv", header=None)

input_tensor = torch.tensor(dataset.values, dtype=torch.float32, requires_grad=True).unsqueeze(0)
#target_tensor = torch.tensor(target_dataset.values, dtype=torch.float32, requires_grad=True).unsqueeze(0)

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
        return output

sequence_length = 36
input_size = 36
hidden_size = 256
num_layers = 1
output_size = 36
num_epochs = 100
learning_rate = 0.01

class EulideanLoss(nn.Module):
    def __init__(self):
        super(EulideanLoss, self).__init__

    def forward(self, Q, q):
        return torch.sqrt(torch.sum((Q-q)**2))

model = RNN(input_size, hidden_size, num_layers, output_size)
criterion = EulideanLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

all_outputs = []

for epoch in range(num_epochs):
    model.train()
    outputs = model(input_tensor)
    q = torch.mean(torch.stack(all))
    loss = criterion(outputs, input_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


    all_outputs.append(outputs.detach().numpy())

    print(f"Epoch: {epoch+1}/{num_epochs}, Loss: {loss.item()}")

all_outputs = np.concatenate(all_outputs, axis=0)
pd.DataFrame(all_outputs).to_csv('output.csv', index=False,header=False)


