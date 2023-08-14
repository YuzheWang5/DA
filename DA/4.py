import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

dataset = pd.read_csv('Q.csv', header=None)
input_tensor = torch.tensor(dataset.values, dtype=torch.float32).unsqueeze(0)

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size): #初始化
        super(RNN, self).__init__()
        self.hidden_size = hidden_size #定义隐藏层维度
        self.num_layers = num_layers #定义隐藏层层数
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True) #batch_first=True：将batch_size作为第一个维度
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device) #初始化h0，即创建一个全0的张量x，维度是（num_layers，batch_size，hidden_size），即（3，1，128）
        if x.dim() == 4:
            x = x.squeeze(2) #检查x的维度，若为4，则压缩至3，符合RNN输入数据的维度
        output, _ = self.rnn(x,h0) #这行代码将输入张量x和初始隐藏状态 h0 传递给RNN模型，并计算RNN的输出。output是RNN在每个时间步的输出，_ 是RNN最后一个时间步的隐藏状态，因为在此并不需要使用最后一个隐藏状态
        output = self.fc(output[:, -1, :]) #这行代码使用全连接层 self.fc 对 RNN 输出进行线性变换，将其转换为输出维度大小的张量
        output = torch.sigmoid(output) #这行代码将线性变换后的张量通过 sigmoid 函数进行激活，将输出限制在 0 到 1 的范围内
        output = (output > 0.5).float() #将结果进行二元化处理
        return output

sequence_length = 36
input_size = 36
hidden_size = 128
num_layers = 3
output_size = 36
num_epochs = 10
learning_rate = 0.01

rnn_model = RNN(input_size, hidden_size, num_layers, output_size)
#criterion = nn.MSELoss()
#optimizer = optim.Adam(rnn_model.parameters(), lr=0.001)

    #loss = criterion(output, input_tensor.unsqueeze(0))
    #optimizer.zero_grad()
    #loss.backward()
    #optimizer.step()

output = rnn_model(input_tensor.unsqueeze(0)) #将输入张量input_tensor传递给RNN模型rnn_model进行前向传播，并将输出结果赋值给变量output
output_array = output.detach().numpy() #将输出张量转化为数组
output_array_t = np.transpose(output_array) #转置输出数组

output_df = pd.DataFrame(output_array_t)
output_df.to_csv('output4.csv', header=None, index=None)