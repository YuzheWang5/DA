import pandas as pd
import torch
import torch.nn as nn

dataset = pd.read_csv('Q.csv', header=None)
input_tensor = torch.tensor(dataset.values, dtype=torch.float32)

class RNN(nn.Module):
    def __init__(self):
        super(RNN,self).__init__()
        self.hidden_size = 128
        self.rnn = nn.RNN(input_size = 1, hidden_size = self.hidden_size)
        self.fc = nn.Linear(self.hidden_size, 1)

    def forward(self, Q):
        Q = self.rnn(Q)
        return Q()

    def init_hidden(self, batch_size):
        hidden = torch.zeros(1, batch_size, self.hidden_size)
        return hidden

rnn_model = RNN()

output = rnn_model(input_tensor)
output_array = output.detach().numpy()
output_df = pd.DataFrame(output_array)
output_df.to_csv('q.csv', header=None, index=None)