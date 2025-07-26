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
        self.device = 'cpu'#device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # weight matrix (x*y, input_len)
        self.weights = torch.rand(x * y, input_len, device=self.device)
        self.locations = self._get_locations().to(self.device)

    def _get_locations(self):
        locs = []
        for i in range(self.x):
            for j in range(self.y):
                locs.append([i, j])
        return torch.tensor(locs, dtype=torch.float)

    def _neighborhood(self, bmu_idx):
        bmu_loc = self.locations[bmu_idx]
        dists = torch.sum((self.locations - bmu_loc) ** 2, dim=1)
        return torch.exp(-dists / (2 * self.sigma ** 2))

    def _find_bmu(self, sample):
        dists = torch.norm(self.weights - sample, dim=1)
        return torch.argmin(dists)

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
            sample = data[torch.randint(0, data.shape[0], (1,)).item()]
            bmu_idx = self._find_bmu(sample)
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
        data = torch.tensor(data, dtype=torch.float, device=self.device)
        bmu_indices = []
        for sample in data:
            bmu_idx = self._find_bmu(sample)
            bmu_indices.append(bmu_idx)
        return self.weights[torch.tensor(bmu_indices, device=self.device)]
    
    def winner(self, x):
        dists = torch.norm(self.weights - x, dim=1)
        bmu_idx = torch.argmin(dists)
        i, j = divmod(bmu_idx.item(), self.y) 
        return i, j
    
    def quantization_error(self, data):
        error = 0.0
        for sample in data:
            bmu_idx = self._find_bmu(sample)
            error += torch.dist(sample, self.weights[bmu_idx])
        return error.item() / len(data)


def som_train(data, x=10, y=10, sigma=1.0, learning_rate=0.05, iters=10000, device="cuda"):
    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    input_len = data.shape[1]
    som = TorchSOM(x, y, input_len, sigma, learning_rate, device=device)

    som.train(data, num_iterations=iters)
    return som

def som_pred(som_model, data, outlier_percentage=0.1):
    data_np = data.cpu().numpy() if isinstance(data, torch.Tensor) else data
    q = som_model.quantize(data_np).detach().cpu().numpy()
    quantization_errors = np.linalg.norm(q - data_np, axis=1)
    error_threshold = np.percentile(quantization_errors, 100 * (1 - outlier_percentage) + 5)
    is_anomaly = quantization_errors > error_threshold
    return is_anomaly.astype(int)

