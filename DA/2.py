import pandas as pd
import torch
import torch.nn as nn
import numpy as np

dataset = pd.read_csv('Q.csv', header=None)
input_tensor = torch.tensor(dataset.values, dtype=torch.float32)

class FC(nn.Module):
    def __init__(self):
        super(FC, self).__init__()
        self.dis = nn.Sequential(
            nn.Linear(12, 144),
            nn.ReLU(True),
            nn.Linear(144, 256),
            nn.ReLU(True),
            nn.Linear(256,12),
            nn.Softmax(dim=1),
        )

    def forward(self, Q):
        Q = self.dis(Q)
        return Q

fc_model = FC()
epochs = 50
criterion = nn.CrossEntropyLoss()
optimizier = torch.optim.Adam(fc_model.parameters(), lr=0.01)

def train():
    for j in range(0, epochs):
        #for i, data in enumerate(input_tensor, 0):
            optimizier.zero_grad()
        
            output = fc_model(input_tensor)
            loss = criterion(output, torch.argmax(input_tensor, dim=1))

            loss.backward()
            optimizier.step()

            if (j + 1) % 10 == 0:
                print('Epoch [{}/{}],loss: {:.6f},'.format(
                j + 1, epochs, loss.item(), ))
                
            torch.save(fc_model.state_dict(), r"C:\Users\woshi\Documents\wyz.pth")
            np.savetxt('MyResult.csv', output.detach().numpy(), fmt='%.2f', delimiter=',')

if __name__ == "__main__":
    train()