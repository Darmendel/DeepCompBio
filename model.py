import torch.nn as nn
import torch.nn.functional as F
from torch import sigmoid


class CNNRBAModel(nn.Module):
    def __init__(self):
        super(CNNRBAModel, self).__init__()

        # Convolutional layer.
        self.conv_layer = nn.Conv1d(4, 700, 5, stride=1, padding=2)

        # Pooling and input dimension.
        self.pooling = nn.MaxPool1d(kernel_size=(40,), stride=(40,))

        # Hidden layers and output layer.
        self.hidden_layer = nn.Linear(700, 700)
        self.fc_out = nn.Linear(700, 1)

    def forward(self, x):
        # Convolutional layer.
        x = x.transpose(-1, -2)  # Transpose dimensions for convolution
        x = F.relu(self.conv_layer(x))
        x = self.pooling(x)
        x = x.view(x.size(0), -1)

        # Hidden layers.
        x = F.relu(self.hidden_layer(x))

        # Output layer.
        return sigmoid(self.fc_out(x))
