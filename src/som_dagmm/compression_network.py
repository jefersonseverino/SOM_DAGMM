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
        self.encoder = nn.Sequential(nn.Linear(self.size, 64),
                                     nn.BatchNorm1d(64),
                                     ResidualBlock(64),
                                     nn.ReLU(),
                                     nn.Linear(64, 2)
                                     )
        self.decoder = nn.Sequential(nn.Linear(2, 64),
                                     nn.BatchNorm1d(64),
                                     ResidualBlock(64),
                                     nn.ReLU(),
                                     nn.Linear(64, self.size))

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
    
class ResidualBlock(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(size, size),
            nn.BatchNorm1d(size),
            nn.ReLU()
        )
    
    def forward(self, x):
        return x + self.block(x)  # Conexão residual
    
class VAENetwork(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(size, 150),
            nn.BatchNorm1d(150),
            nn.ReLU(),
            nn.Linear(150, 75),
            nn.BatchNorm1d(75),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(75, 2)    # Média
        self.fc_var = nn.Linear(75, 2)   # Variância

        self.decoder = nn.Sequential(
            nn.Linear(2, 75),
            nn.BatchNorm1d(75),
            nn.ReLU(),
            nn.Linear(75, 150),
            nn.BatchNorm1d(150),
            nn.ReLU(),
            nn.Linear(150, size)
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_var(h)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = mu + torch.randn_like(logvar) * torch.exp(0.5*logvar)
        return self.decode(z), mu, logvar
    
class SkipCompressionNetwork(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.enc1 = nn.Linear(size, 150)
        self.enc2 = nn.Linear(150, 75)
        self.enc3 = nn.Linear(75, 2)
        
        self.dec1 = nn.Linear(2, 75)
        self.dec2 = nn.Linear(75, 150)
        self.dec3 = nn.Linear(150, size)
        
    def forward(self, x):
        # Encoder
        h1 = torch.relu(self.enc1(x))
        h2 = torch.relu(self.enc2(h1))
        z = self.enc3(h2)
        
        # Decoder com skip connections
        d1 = torch.relu(self.dec1(z) + h2)  # Skip
        d2 = torch.relu(self.dec2(d1) + h1) # Skip
        return self.dec3(d2)