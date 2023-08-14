import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Generated Q martix randomly
Q = np.zeros((684,1))
segment_sum = 2
for i in range(57):
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
        output = torch.nn.functional.gumbel_softmax(output, hard=True)
        #output2 = (output > 0.5).float()  # binary output
        return output.unsqueeze(-1)


sequence_length = 684
input_size = 1
hidden_size = 256
num_layers = 1
output_size = 684
num_epochs = 100
learning_rate = 0.01

class EulideanLoss(nn.Module):
    def __init__(self):
        super(EulideanLoss, self).__init__()

    def forward(self, x, y):
        return torch.sqrt(torch.sum((x-y)**2))

model = RNN(input_size, hidden_size, num_layers, output_size)
criterion = EulideanLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

losses = []
qn = torch.zeros_like(input_tensor)

for epoch in range(num_epochs):
    model.train()
    Sn = model(input_tensor)
    #Sn_binary = (Sn > 0.5).float()
    qn_new = (qn.detach() * epoch + Sn) / (epoch + 1)
    loss = criterion(qn_new, input_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

    print(f"Epoch: {epoch+1}/{num_epochs}, Loss: {loss.item()}")

    qn = qn_new.detach()

plt.plot(losses)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Loss Function')
plt.show()

# Save the final qn to CSV
qn_np = qn.squeeze(0).numpy()
pd.DataFrame(qn_np).to_csv('output.csv', index=False,header=False)