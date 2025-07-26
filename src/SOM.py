import torch
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import StandardScaler

class TorchSOM:
    def __init__(self, x, y, input_len, sigma=1.0, learning_rate=0.05, device=None):
        self.x = x
        self.y = y
        self.input_len = input_len
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # weight matrix (x*y, input_len)
        self.weights = torch.rand(x * y, input_len, device=self.device)
        self.locations = self._get_locations().to(self.device)

    def _get_locations(self):
        i = torch.arange(self.x, dtype=torch.float32)
        j = torch.arange(self.y, dtype=torch.float32)
        grid_x, grid_y = torch.meshgrid(i, j, indexing='ij')
        return torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)

    def _neighborhood(self, bmu_idx):
        bmu_loc = self.locations[bmu_idx]
        dists = torch.sum((self.locations - bmu_loc) ** 2, dim=1)
        return torch.exp(-dists / (2 * self.sigma ** 2))

    def _find_bmu_batch(self, data):
        # data: (batch_size, input_len)
        dists = torch.cdist(data, self.weights, p=2)  # (batch_size, n_nodes)
        bmu_indices = torch.argmin(dists, dim=1)
        return bmu_indices

    def train(self, data, num_iterations=10000):
        count_trys = 0
        MAX_TRYS = 5
        best_weights = self.weights.clone()
        best_error = float('inf')
        data = torch.tensor(data, dtype=torch.float, device=self.device)

        lr0 = self.learning_rate
        sigma0 = self.sigma
        sigma_min = 0.01

        decay = 1 
        curr_lr = lr0 * decay
        curr_sigma = sigma_min + (sigma0 - sigma_min) * decay
        
        for i in range(num_iterations):
            print(f"Treinamento SOM: Iteração:{i} de {num_iterations}")
            sample = data[torch.randint(0, data.size(0), (1,), device=self.device)]
            bmu_idx = self._find_bmu_batch(sample)[0]
            bmu_loc = self.locations[bmu_idx]
            dists = torch.sum((self.locations - bmu_loc) ** 2, dim=1)
            theta = torch.exp(-dists / (2 * curr_sigma ** 2)).unsqueeze(1)
            delta = curr_lr * theta * (sample - self.weights)
            self.weights += delta
            
            q_error = self.quantization_error(data)
            if q_error < best_error:
                print(f"    Novo melhor erro: {q_error}")
                decay = 1 * 0.9
                best_error = q_error
                best_weights = self.weights.clone()
                count_trys = 0
            else:
                count_trys += 1
                print(f"    Sem melhoras. Tentativa: {count_trys} de {MAX_TRYS}")
                if count_trys >= MAX_TRYS:
                    print(f"Convergência após {i} iterações.")
                    break

            
        
        # Ao fim do treinamento, substitua pelos melhores pesos
        self.weights = best_weights

    def quantize(self, data):
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32, device=self.device)
        data = data.to(self.device)
        bmu_indices = self._find_bmu_batch(data)
        return self.weights[bmu_indices]

    
    def winner(self, x):
        x = x.to(self.device)
        dists = torch.norm(self.weights - x, dim=1)
        bmu_idx = torch.argmin(dists)
        return divmod(bmu_idx.item(), self.y)
    
    def quantization_error(self, data):
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)
        data = data.to(self.device)
        q = self.quantize(data)
        return torch.mean(torch.norm(data - q, dim=1)).item()


def som_train(data, x=10, y=10, sigma=1.0, learning_rate=0.05, iters=10000, device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    som = TorchSOM(x, y, data.shape[1], sigma, learning_rate, device=device)
    som.train(data, num_iterations=iters)
    return som


def som_pred(som_model, data, outlier_percentage=0.1):
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data, dtype=torch.float32)
    data = data.to(som_model.device)

    q = som_model.quantize(data)
    errors = torch.norm(data - q, dim=1)
    threshold = torch.quantile(errors, 1 - outlier_percentage + 0.05)
    return (errors > threshold).int().cpu().numpy()

