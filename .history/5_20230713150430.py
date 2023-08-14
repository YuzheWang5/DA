import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

dataset = pd.read_csv("Q.csv", header=None)
#target_dataset = pd.read_csv("Q_target.csv", header=None)

input_tensor = torch.tensor(dataset.values, dtype=torch.float32).unsqueeze(0)
#target_tensor = torch.tensor(target_dataset.values, dtype=torch.float32).unsqueeze(0)


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        if x.dim() == 4:
            x = x.squeeze(2)
        output, _ = self.rnn(x, h0)
        output = self.fc(output[:, -1, :])
        output = torch.sigmoid(output)
        output = (output > 0.5).float()
        return output


sequence_length = 36
input_size = 36
hidden_size = 256
num_layers = 3
output_size = 36
num_epochs = 10
learning_rate = 0.01


class EuclideanLoss(nn.Module):
    def __init__(self):
        super(EuclideanLoss, self).__init__()

    def forward(self, Q, q):
        return torch.sqrt(torch.sum((Q-q)**2))


rnn_model = RNN(input_size, hidden_size, num_layers, output_size)
criterion = EuclideanLoss()
optimizer = torch.optim.Adam(rnn_model.parameters(), lr=learning_rate)

all_outputs = []

for epoch in range(num_epochs):
    # Forward pass
    outputs = rnn_model(input_tensor)
    all_outputs.append(outputs.detach())

    q = torch.mean(torch.stack(all_outputs),dim=0)
    loss = criterion(input_tensor, q)

    # Backward and optimize
    optimizer.zero_grad()
    #loss.backward()
    optimizer.step()

    print("Epoch [{}/{}], Loss: {:.4f}".format(epoch + 1, num_epochs, loss.item()))

    output = rnn_model(input_tensor.unsqueeze(0))
    output_array = outputs.detach().numpy()
    output_array_t = np.transpose(output_array)
    #all_outputs.append(output_array_t)
    all_outputs.append(outputs.detach())

final_output = np.hstack(all_outputs)

final_df = pd.DataFrame(final_output)
final_df.to_csv("output5.csv", header=None, index=None)
