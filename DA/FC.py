import pandas as pd
import torch
import torch.nn as nn

class FC(nn.Module):
    def __init__(self):
        super(FC, self).__init__()
        self.fc = nn.Linear(684, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, Q):
        Q = Q.transpose(0, 1)
        out = self.fc(Q)
        out = self.sigmoid(out)
        return out.round()

fc_model = FC()

input_data = pd.read_csv('Q.csv', header=None)
input_tensor = torch.tensor(input_data.values, dtype=torch.float32)

output = fc_model(input_tensor)
output_array = output.detach().numpy()
output_df = pd.DataFrame(output_array)
output_df.to_csv('q.csv', header=None, index=None)