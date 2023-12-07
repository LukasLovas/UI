import torch.nn as nn


class Feedforward_model_relu(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Feedforward_model_relu, self).__init__()
        self.input_layer = nn.Linear(input_dim, 50)
        self.hidden_layer = nn.Linear(50, 50)
        self.output_layer = nn.Linear(50, output_dim)
        self.activation_func = nn.ReLU()

    def forward(self, x):
        x = self.input_layer(x)
        x = self.activation_func(self.hidden_layer(x))
        out = self.activation_func(self.output_layer(x))
        return out
