import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import torch.optim as optim

dataset = pd.read_csv('Q.csv', header=None)
input_tensor = torch.tensor(dataset.values, dtype=torch.float32).unsqueeze(0)

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
        out, _ = self.rnn(x,h0)
        out = self.fc(out[:, -1, :])
        out = torch.sigmoid(out)
        return out

sequence_length = 36
input_size = 36
hidden_size = 128
num_layers = 3
output_size = 36
num_epochs = 10
learning_rate = 0.01
#n_iter =  5

model = RNN(input_size, hidden_size, num_layers, output_size)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Training
for epoch in range(num_epochs):
    model.train()
    output = model(input_tensor)
    #for _ in range(n_iter):
        #output = model(input_tensor)
        #outputs.append(output) #

    #outputs = torch.stack(outputs)
    #outputs = outputs.mean(dim=0)

    # Compute loss
    loss = criterion(output, input_tensor.squeeze())

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 100 == 0:
        print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

# Save model
torch.save(model.state_dict(), 'model.ckpt')

# Test
model.eval()
#test_outputs = []
with torch.no_grad():
    #for _ in range(n_iter):
        #output = model(input_tensor)
        #test_outputs.append(output)
#test_outputs = torch.stack(test_outputs)
     predictions = model(input_tensor) #test_outputs.mean(dim=0)

# Post-process predictions
predictions = predictions.detach().numpy()
predictions = np.round(predictions).astype(int)

# Save output
np.savetxt("q.csv", predictions, delimiter=",")
