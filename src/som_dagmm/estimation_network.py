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



class EstimationNetwork(nn.Module):
    """Defines a estimation network."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
                                    nn.Linear(6, 32),  # Aumentei a primeira camada
                                    nn.BatchNorm1d(32),
                                    nn.ReLU(),
                                    nn.Dropout(0.3),   # Reduzi dropout
                                    nn.Linear(32, 256), # Camadas mais largas
                                    nn.BatchNorm1d(256),
                                    nn.ReLU(),
                                    nn.Linear(256, 128),
                                    nn.BatchNorm1d(128),
                                    nn.ReLU(),
                                    nn.Linear(128, 2),  # Saída direta sem camadas extras
                                    nn.Softmax(dim=1)
                                )
    def forward(self, input):
        return self.net(input)
    
class EstimationNetworkWithAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_net = nn.Sequential(
            nn.Linear(6, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.attention = nn.MultiheadAttention(embed_dim=64, num_heads=2)
        self.output_net = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.base_net(x)
        x = x.unsqueeze(0)  # Shape: [1, batch_size, 64]
        x, _ = self.attention(x, x, x)
        x = x.squeeze(0)
        return self.output_net(x)