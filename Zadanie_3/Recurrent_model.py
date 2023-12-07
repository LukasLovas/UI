import torch
import torch.nn as nn


class Recurrent_model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Recurrent_model, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(x.size(0), self.hidden_size).to(x.device)
        h0 = h0.unsqueeze(0).expand(1, -1, -1)  # Unsqueeze for 3d
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])  # Last time step
        return out
