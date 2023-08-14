import pandas as pd
import torch
import torch.nn as nn

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
            nn.Linear(256,684),
        )

    def forward(self, Q):
        Q = Q.transpose(0, 1)
        return Q.round()

fc_model = FC()

output = fc_model(input_tensor)
output_array = output.detach().numpy()
output_df = pd.DataFrame(output_array)
output_df.to_csv('output3.csv', header=None, index=None)