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
        self.net = nn.Sequential(nn.Linear(6, 10),
                                 nn.BatchNorm1d(10),
                                 nn.Tanh(),
                                 nn.Dropout(p=0.5),
                                 nn.Linear(10, 150),
                                 nn.BatchNorm1d(150),
                                 nn.ReLU(),
                                 nn.Linear(150, 300),
                                 nn.BatchNorm1d(300),
                                 nn.ReLU(),
                                 nn.Linear(300, 150),
                                 nn.BatchNorm1d(150),
                                 nn.ReLU(),
                                 nn.Linear(150, 10),
                                 nn.BatchNorm1d(10),
                                 nn.ReLU(),
                                 nn.Linear(10, 2),
                                 nn.Softmax(dim=1)
                                 )
    def forward(self, input):
        return self.net(input)