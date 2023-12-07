import torch.nn as nn
import torch.nn.functional as f

class Feedforward_model_mix(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Feedforward_model_mix, self).__init__()
        self.input_layer = nn.Linear(input_dim, 64)
        self.hidden_layer1 = nn.Linear(64, 32)
        self.hidden_layer2 = nn.Linear(32, 16)
        self.output_layer = nn.Linear(16, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.input_layer(x)
        x = self.relu(self.hidden_layer1(x))
        x = self.relu(self.hidden_layer2(x))
        out = f.softmax(self.output_layer(x), dim=1)
        return out
