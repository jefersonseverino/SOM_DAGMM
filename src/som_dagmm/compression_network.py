"""Defines the compression network."""

import torch, random, os
from torch import nn

SEED = 42

# Python & libs padrão
random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

# Torch CPU & GPU
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)  # para multi-GPU



class CompressionNetwork(nn.Module):
    """Defines a compression network."""
    def __init__(self, size):
        super().__init__()
        self.size = size
        print(size)
        self.encoder = nn.Sequential(nn.Linear(self.size, 150),
                                     nn.BatchNorm1d(150),
                                     nn.ReLU(),
                                     nn.Linear(150, 75),
                                     nn.BatchNorm1d(75),
                                     nn.ReLU(),
                                     nn.Linear(75, 10),
                                     nn.BatchNorm1d(10),
                                     nn.ReLU(),
                                     nn.Linear(10, 2)
                                     )
        self.decoder = nn.Sequential(nn.Linear(2, 10),
                                     nn.BatchNorm1d(10),
                                     nn.ReLU(),
                                     nn.Linear(10, 75),
                                     nn.BatchNorm1d(75),
                                     nn.ReLU(),
                                     nn.Linear(75, 150),
                                     nn.BatchNorm1d(150),
                                     nn.ReLU(),
                                     nn.Linear(150, self.size))

        self._reconstruction_loss = nn.MSELoss()

    def forward(self, input):
        out = self.encoder(input)
        out = self.decoder(out)

        return out

    def encode(self,  input):
        return self.encoder(input)

    def decode(self, input):
        return self.decoder(input)

    def reconstruction_loss(self, input):
        target_hat = self(input)
        return self._reconstruction_loss(target_hat, input)