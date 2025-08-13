"""Implements a GMM model."""

import torch, random, os
import numpy as np
from torch import nn

SEED = 42

# Python & libs padrão
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

# Torch CPU & GPU
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)  # para multi-GPU


class GMM(nn.Module):
    """Implements a Gaussian Mixture Model."""
    def __init__(self, num_mixtures, dimension_embedding):
        """Creates a Gaussian Mixture Model.

        Args:
            num_mixtures (int): the number of mixtures the model should have.
            dimension_embedding (int): the number of dimension of the embedding
                space (can also be thought as the input dimension of the model)
        """
        super().__init__()
        self.num_mixtures = num_mixtures
        self.dimension_embedding = dimension_embedding

        self.mixtures = nn.ModuleList([Mixture(dimension_embedding) for _ in range(num_mixtures)])

    def forward(self, inputs):
        all_outputs = torch.stack([m(inputs, with_log=False) for m in self.mixtures])
        out = all_outputs.sum(dim=0)
        return -torch.log(out)

    def _update_mixtures_parameters(self, samples, mixtures_affiliations):
        if not self.training:
            return

        # Transpose once for column-wise access (if num_mixtures is large)
        affiliations_t = mixtures_affiliations.t()  # [num_mixtures, batch_size]
        
        for i, mixture in enumerate(self.mixtures):
            # Direct column access from pre-transposed tensor
            mixture._update_parameters(samples, affiliations_t[i])



class Mixture(nn.Module):
    def __init__(self, dimension_embedding):
        super().__init__()
        self.dimension_embedding = dimension_embedding
        self.Phi = nn.Parameter(torch.rand(1), requires_grad=False)
        self.mu = nn.Parameter(2 * torch.rand(dimension_embedding) - 0.5, 
                            requires_grad=False)
        self.Sigma = nn.Parameter(torch.eye(dimension_embedding),
                                requires_grad=False)
        self.eps_Sigma = torch.diag(torch.full((dimension_embedding,), 1e-8))


    def forward(self, samples, with_log=True):
        """Samples has shape [batch_size, dimension_embedding]"""
        # TODO: cache the matrix inverse and determinant?
        # TODO: so ugly and probably inefficient: do we have to create those
        #       new variables and conversions from numpy?
        batch_size, _ = samples.shape
        out_values = []
        inv_sigma = torch.pinverse(self.Sigma)
        det_sigma = np.linalg.det(self.Sigma.data.cpu().numpy())
        det_sigma = torch.from_numpy(det_sigma.reshape([1])).float()
        det_sigma = torch.autograd.Variable(det_sigma)
        for sample in samples:
            diff = (sample - self.mu).view(-1, 1)
            #det_sigma = torch.from_numpy(det_sigma).float()

            out = -0.5 * torch.mm(torch.mm(diff.view(1, -1), inv_sigma), diff)
            out = (self.Phi * torch.exp(out)) / (torch.sqrt(2. * np.pi * det_sigma))
            if with_log:
                out = -torch.log(out)
            out_values.append(float(out.data.cpu().numpy()))

        out = torch.autograd.Variable(torch.FloatTensor(out_values))
        return out

    def _update_parameters(self, samples, affiliations):
        if not self.training:
            return

        # Updating phi - versão vetorizada
        self.Phi.data = torch.mean(affiliations).data

        # Updating mu - versão vetorizada
        weighted_samples = samples * affiliations.view(-1, 1)
        self.mu.data = (weighted_samples.sum(dim=0) / (affiliations.sum() + 1e-12)).data

        # Updating Sigma - versão vetorizada
        diff = samples - self.mu.unsqueeze(0)
        weighted_diff = diff * torch.sqrt(affiliations).view(-1, 1)  # sqrt para estabilidade numérica
        num = weighted_diff.t() @ weighted_diff
        denom = affiliations.sum() + 1e-12
        self.Sigma.data = (num / denom).data + self.eps_Sigma.to(num.device)

    def gmm_loss(self, out, L1, L2):
        term1 = L1 * out.mean()
        
        sigma_diag = torch.diagonal(self.Sigma)
        term2 = L2 * torch.sum(1.0 / (sigma_diag + 1e-12))
        
        return term1 + term2