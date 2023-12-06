import torch.nn as nn


class Classification_model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Classification_model, self).__init__()
        self.input_layer = nn.Linear(input_dim, 50)
        self.hidden_layer1 = nn.Linear(50, 50)
        self.output_layer = nn.Linear(50, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.input_layer(x))
        x = self.relu(self.hidden_layer1(x))
        out = self.output_layer(x)
        return out
