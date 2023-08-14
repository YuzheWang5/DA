import pandas as pd
import torch
import torch.nn as nn

dataset = pd.read_csv('W.csv', header=None)
input_tensor = torch.tensor(dataset.values, dtype=torch.float32)

class FCNN(nn.Module):
    def __init__(self):
        super(FCNN, self).__init__()
        self.dis = nn.Sequential(
        nn.Sigmoid()
        )

    def forward(self, W):
        W = self.dis(W)
        return W.round()

fc_model = FCNN()

output = fc_model(input_tensor)
output_array = output.detach().numpy()
output_df = pd.DataFrame(output_array)
output_df.to_csv('b.csv', header=None, index=None)