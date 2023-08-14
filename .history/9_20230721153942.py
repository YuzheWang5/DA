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
        output = torch.nn.functional.gumbel_softmax(output, hard=True)  # Gumbel-Softmax
        return output.unsqueeze(-1)
