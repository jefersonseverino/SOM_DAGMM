import os
import pickle
import gc
import sys
import argparse
import numpy as np

import torch 
from utils import *

from minisom import MiniSom
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

from som_dagmm.model import DAGMM, SOM_DAGMM
from som_dagmm.compression_network import CompressionNetwork
from som_dagmm.estimation_network import EstimationNetwork
from som_dagmm.gmm import GMM, Mixture

from SOM import som_train, som_pred
from sklearn.model_selection import train_test_split
import tqdm, copy
import random

SEED = 10
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def parse_args():
    parser = argparse.ArgumentParser(description='Anomaly Detection with unsupervised methods')
    parser.add_argument('--dataset', dest='dataset', help='training dataset', default='vehicle_claims', type=str)
    parser.add_argument('--embedding', dest='embed', help='one_hot, label', default='NULL', type=str)
    parser.add_argument('--features', dest='features', help='all, numerical, categorical', default='all', type=str)
    parser.add_argument('--batch_size', dest='batch_size', help='32', default = 32, type=int)
    parser.add_argument('--epoch', dest='epoch', help='1', default='1', type=int)
    parser.add_argument('--contamination', dest='contamination', help='0.1', default=0.0, type=float)
    args = parser.parse_args()
    return args

args = parse_args()
epochs = args.epoch
batch_size = args.batch_size
save_path = os.path.join(args.dataset + "_" + args.features + "_" + args.embed + "_" + str(args.contamination) + ".pt")

names = [i for i in range(0,43)]
data = load_data('data/NSL-KDD/KDDTrain+.txt', names)
categorical_cols = [1,2,3]
 
#encode categorical variables 
if args.embed == 'one_hot':
    data = one_hot_encoding(data, categorical_cols)
if args.embed == 'label_encode':
    data = label_encoding(data, categorical_cols)

data_benign = data[data[41] == "normal"].copy()
data_malicious = data[data[41] != "normal"].copy()
Y_benign, data_benign = get_labels(data_benign, args.dataset)
Y_malicious, data_malicious = get_labels(data_malicious, args.dataset)

# Remove columns with NA values
data_benign = fill_na(data_benign)
data_malicious = fill_na(data_malicious)

# normalize data
data_benign = normalize_cols(data_benign)
data_malicious = normalize_cols(data_malicious)

#test and train split
train_data, val_data, Y_train, Y_val = train_test_split(data_benign, Y_benign, test_size=0.5, random_state=SEED)
val_data, test_data, Y_val, Y_test = train_test_split(val_data, Y_val, test_size=0.5, random_state=SEED)
val_mal_data, test_mal_data, Y_val_mal, Y_test_mal = train_test_split(data_malicious, Y_malicious, test_size=0.5, random_state=SEED)

val_data = pd.concat([val_data, val_mal_data], ignore_index=True)
test_data = pd.concat([test_data, test_mal_data], ignore_index=True)

# Concatenate Y
Y_val = np.concatenate([Y_val, Y_val_mal])
Y_test = np.concatenate([Y_test, Y_test_mal])

if args.contamination > 0:
    train_data = train_data.reset_index(drop=True)
    val_data = val_data.reset_index(drop=True)
    
    malicious_index = np.where(Y_val == 1)[0]
    benign_index = np.where(Y_train == 0)[0]
    n_samples = int(len(train_data) * args.contamination)
    selected_malicious_index = np.random.choice(malicious_index, n_samples, replace=False)
    selected_benign_index = np.random.choice(benign_index, n_samples, replace=False)
    
    benign_data = train_data.iloc[selected_benign_index]
    malicious_data = val_data.iloc[selected_malicious_index]

    train_data = train_data.drop(index=selected_benign_index).reset_index(drop=True)
    val_data = val_data.drop(index=selected_malicious_index).reset_index(drop=True)
    
    Y_train = np.delete(Y_train, selected_benign_index)
    Y_val = np.delete(Y_val, selected_malicious_index)

    train_data = pd.concat([train_data, malicious_data], ignore_index=True)
    Y_train = np.concatenate([Y_train, np.ones(len(malicious_data))])

    val_data = pd.concat([val_data, benign_data], ignore_index=True)
    Y_val = np.concatenate([Y_val, np.zeros(len(benign_data))])

#train_data = train_data.values.astype(np.float32)
print(train_data.shape)

#Convert to torch tensors
train_data = torch.tensor(train_data.values.astype(np.float32))
val_data = torch.tensor(val_data.values.astype(np.float32))
test_data = torch.tensor(test_data.values.astype(np.float32))

#Convert tensor to TensorDataset class.
train_dataset = TensorDataset(train_data)
val_dataset = TensorDataset(val_data)
test_dataset = TensorDataset(test_data)

#TrainLoader
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

compression = CompressionNetwork(train_data.shape[1])
estimation = EstimationNetwork()
gmm = GMM(2,6)
mix = Mixture(6)
dagmm = DAGMM(compression, estimation, gmm)

train_np = train_data.numpy()
# Use pretrained SOM
som = som_train(train_np, x=10, y=10, sigma=1, learning_rate=0.7, iters=10000)

net = SOM_DAGMM(som, dagmm)
optimizer =  optim.Adam(net.parameters(), lr=1e-4)

OFFSET = 0.2
MAX_ATTEMPT = 5
best_val_loss = 100
count_since_better_loss = 0

for epoch in range(epochs):
    print('EPOCH {}:'.format(epoch + 1))
    running_loss = 0
    for i, data in enumerate(tqdm.tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")):
        out = net(data[0])
        optimizer.zero_grad()
        L_loss = compression.reconstruction_loss(data[0])
        G_loss = mix.gmm_loss(out=out, L1=0.1, L2=0.005)
        loss = L_loss + G_loss
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    net.eval()
    val_loss = 0
    with torch.no_grad():
        for val_batch in val_dataloader:
            val_input = val_batch[0]
            val_out = net(val_input)
            val_L_loss = compression.reconstruction_loss(val_input)
            val_G_loss = mix.gmm_loss(out=val_out, L1=0.1, L2=0.005)
            val_total_loss = val_L_loss + val_G_loss
            val_loss += val_total_loss.item() if torch.isfinite(val_total_loss) else 0
    print(f"Validation - Epoch {epoch+1} - Loss: {val_loss:.4f}")

    if val_loss < best_val_loss - OFFSET:
        print(f"New best validation loss: {val_loss:.4f}")
        best_val_loss = val_loss
        count_since_better_loss = 0
        best_net = copy.deepcopy(net.state_dict())
    else:
        count_since_better_loss += 1
        print(f"    Sem melhoras. Tentativa: {count_since_better_loss} de {MAX_ATTEMPT}")
        if count_since_better_loss >= MAX_ATTEMPT:
            print(f"Convergência após {epoch} iterações.")
            break
    
    net.train()

net.load_state_dict(best_net)
net.train()
torch.save(net, save_path)