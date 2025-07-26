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
import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Anomaly Detection with unsupervised methods')
    parser.add_argument('--dataset', dest='dataset', help='training dataset', default='vehicle_claims', type=str)
    parser.add_argument('--embedding', dest='embed', help='one_hot, label', default='NULL', type=str)
    parser.add_argument('--features', dest='features', help='all, numerical, categorical', default='all', type=str)
    parser.add_argument('--batch_size', dest='batch_size', help='32', default = 32, type=int)
    parser.add_argument('--epoch', dest='epoch', help='1', default='1', type=int)
    args = parser.parse_args()
    return args

args = parse_args()
epochs = args.epoch
batch_size = args.batch_size
save_path = os.path.join(args.dataset + "_" + args.features + "_" + args.embed + ".pt")

names = [i for i in range(0,43)]
data = load_data('data/NSL-KDD/KDDTrain+.txt', names)
categorical_cols = [1,2,3,4]
data_benign = data[data[41] == "normal"].copy()
data_malicious = data[data[41] != "normal"].copy()
Y_benign, data_benign = get_labels(data_benign, args.dataset)
Y_malicious, data_malicious = get_labels(data_malicious, args.dataset)
 
#encode categorical variables 
if args.embed == 'one_hot':
    # data = one_hot_encoding(data, categorical_cols)
    data_benign = one_hot_encoding(data_benign, categorical_cols)
    data_malicious = one_hot_encoding(data_malicious, categorical_cols)
if args.embed == 'label_encode':
    # data = label_encoding(data, categorical_cols)
    data_benign = label_encoding(data_benign, categorical_cols)
    data_malicious = label_encoding(data_malicious, categorical_cols)

# Remove columns with NA values
# data = fill_na(data)
data_benign = fill_na(data_benign)
data_malicious = fill_na(data_malicious)

# normalize data
# data = normalize_cols(data)
data_benign = normalize_cols(data_benign)
data_malicious = normalize_cols(data_malicious)

#test and train split
# train_data, test_data, Y_train, Y_test = train_test_split(data, Y, test_size=0.2)
train_data, val_data, Y_train, Y_val = train_test_split(data_benign, Y_benign, test_size=0.4)
# Split again for validation and test
val_data, test_data, Y_val, Y_test = train_test_split(val_data, Y_val, test_size=0.5)

# Split malicious data only for validation and test
val_mal_data, test_mal_data, Y_val_mal, Y_test_mal = train_test_split(data_malicious, Y_malicious, test_size=0.5)

# Concatenate benign and malicious data for validation and test
val_data = pd.concat([val_data, val_mal_data], ignore_index=True)
test_data = pd.concat([test_data, test_mal_data], ignore_index=True)

# Concatenate Y
Y_val = np.concatenate([Y_val, Y_val_mal])
Y_test = np.concatenate([Y_test, Y_test_mal])

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
som = som_train(train_np, x=10, y=10, sigma=1, learning_rate=0.8, iters=10000)

net = SOM_DAGMM(som, dagmm)
optimizer =  optim.Adam(net.parameters(), lr=1e-4)

best_val_loss = 100
count_since_better_loss = 0
MAX_ATTEMPT = 5
OFFSET = 0.5

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
            val_loss += val_total_loss.item()
    print(f"Validation - Epoch {epoch+1} - Loss: {val_loss:.4f}")

    if val_loss < best_val_loss - OFFSET:
        print(f"New best validation loss: {val_loss:.4f}")
        best_val_loss = val_loss
        count_since_better_loss = 0
        best_net = net
    else:
        count_since_better_loss += 1
        print(f"Trys since better loss: {count_since_better_loss}")
        if count_since_better_loss > MAX_ATTEMPT:
            print(f"Early stop")
            break

    net.train()

torch.save(best_net, save_path)
