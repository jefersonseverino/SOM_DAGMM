"""Implements all the components of the DAGMM model."""

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


eps = torch.tensor(1e-8, dtype=torch.float32)

class SOM_DAGMM(nn.Module):
    def __init__(self, som, dagmm):
        super().__init__()
        self.dagmm = dagmm
        self.som = som

    def forward(self, input):
        winners = torch.tensor(
            [normalize_tuple(self.som.winner(i), 10) for i in input],
            dtype=torch.float32,
            device=input.device
        )
        return self.dagmm(input, winners)
       
        
class DAGMM(nn.Module):
    def __init__(self, compression_module, estimation_module, gmm_module):
        """
        Args:
            compression_module (nn.Module): an autoencoder model that
                implements at leat a function `self.encoder` to get the
                encoding of a given input.
            estimation_module (nn.Module): a FFNN model that estimates the
                memebership of each input to a each mixture of a GMM.
            gmm_module (nn.Module): a GMM model that implements its mixtures
                as a list of Mixture classes. The GMM model should implement
                the function `self._update_mixtures_parameters`.
        """
        super().__init__()

        self.compressor = compression_module
        self.estimator = estimation_module
        self.gmm = gmm_module
        

    def forward(self, input, winners):
        # Forward in the compression network.
        encoded = self.compressor.encode(input)
        decoded = self.compressor.decode(encoded)

        # Preparing the input for the estimation network.
        relative_ed, cosine_sim = relative_and_cosine(input, decoded)
        # Adding a dimension to prepare for concatenation.
        relative_ed = relative_ed.view(-1, 1)
        cosine_sim = cosine_sim.view(-1, 1)
        latent_vectors = torch.cat([encoded, relative_ed, cosine_sim, winners], dim=1)
        # latent_vectors has shape [batch_size, dim_embedding + 2]

        # Updating the parameters of the mixture.
        if self.training:
            mixtures_affiliations = self.estimator(latent_vectors)
            # mixtures_affiliations has shape [batch_size, num_mixtures]
            self.gmm._update_mixtures_parameters(latent_vectors,
                                                 mixtures_affiliations)
        # Estimating the energy of the samples.
        return self.gmm(latent_vectors)


def relative_and_cosine(x1, x2, eps=1e-8):
    dist_x1 = torch.norm(x1, p=2, dim=1)
    dist_x2 = torch.norm(x2, p=2, dim=1)
    rel_ed = torch.norm(x1 - x2, p=2, dim=1) / torch.clamp(dist_x1, min=eps)
    cosine_sim = torch.sum(x1 * x2, dim=1) / torch.clamp(dist_x1 * dist_x2, min=eps)
    return rel_ed.view(-1, 1), cosine_sim.view(-1, 1)

def normalize_tuple(x, norm_val):
    a, b = x
    a = a/norm_val
    b = b/norm_val
    return (a,b)
