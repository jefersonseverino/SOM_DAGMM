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

SEED = 42

# Python & libs padrão
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

# Torch CPU & GPU
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)  # para multi-GPU

def parse_args():
    parser = argparse.ArgumentParser(description='Anomaly Detection with unsupervised methods')
    parser.add_argument('--dataset', dest='dataset', help='training dataset', default='kdd', type=str)
    parser.add_argument('--embedding', dest='embed', help='one_hot, label', default='NULL', type=str)
    parser.add_argument('--features', dest='features', help='all, numerical, categorical', default='all', type=str)
    parser.add_argument('--batch_size', dest='batch_size', help='32', default = 32, type=int)
    parser.add_argument('--epoch', dest='epoch', help='1', default='1', type=int)
    parser.add_argument('--contamination', dest='contamination', help='0.1', default=0.0, type=float)
    parser.add_argument('--seed', dest='seed', help='42', default=42, type=int)
    parser.add_argument('--lrsom', dest='lrsom', help='0.4', default=0.4, type=float)
    parser.add_argument('--nf', dest='nf', help='bubble', default='bubble', type=str)
    parser.add_argument('--lrdagmm', dest='lrdagmm', help='0.0001', default=0.0001, type=float)
    parser.add_argument('--l1', dest='l1', help='0.1', default=0.1, type=float)
    parser.add_argument('--l2', dest='l2', help='0.005', default=0.005, type=float)
    args = parser.parse_args()
    return args

args = parse_args()
epochs = args.epoch
batch_size = args.batch_size
save_path = os.path.join(args.dataset + "_" + args.features + "_" + args.embed + "_" + str(args.contamination) + ".pt")

if args.dataset == 'credit_card':
    data = load_data('data/CreditCardFraud/creditcard.csv')
    Y = get_labels(data, args.dataset)

    data_benign = data[data['Class'] == 0].copy()
    data_malicious = data[data['Class'] == 1].copy()
elif args.dataset == 'arrhythmia':
    names = [i for i in range(1,281)]
    data = load_data('data/arrhythmia.csv', names)
    data = remove_cols(data, 14)
    Y = get_labels(data, args.dataset)

    data_benign = data[data[280] == 1].copy()
    data_malicious = data[data[280] != 1].copy()
elif args.dataset == 'kdd':
    names = [i for i in range(0,43)]
    data = load_data('data/NSL-KDD/KDDTrain+.txt', names)
    categorical_cols = [1,2,3]

    #encode categorical variables 
    if args.embed == 'one_hot':
        data = one_hot_encoding(data, categorical_cols)
    elif args.embed == 'label_encode':
        data = label_encoding(data, categorical_cols)

    data_benign = data[data[41] == "normal"].copy()
    data_malicious = data[data[41] != "normal"].copy()

elif args.dataset == 'cic':
    data = pd.DataFrame()
    for file in os.listdir('data/CIC-2018/CSE-CIC-IDS2018/'):
        if file.endswith('.csv'):
            file_path = os.path.join('data/CIC-2018/CSE-CIC-IDS2018/', file)
            df = pd.read_csv(file_path)
            data = pd.concat([data, df], ignore_index=True)
    data = data.sample(n=500000, random_state=args.seed)
        
    data_benign = data[data['Label'] == 'Benign'].copy()
    data_malicious = data[data['Label'] != 'Benign'].copy()

Y_benign, data_benign = get_labels(data_benign, args.dataset)
Y_malicious, data_malicious = get_labels(data_malicious, args.dataset)

# Remove columns with NA values
data_benign = fill_na(data_benign)
data_malicious = fill_na(data_malicious)

# normalize data
#data_benign = normalize_cols(data_benign)
#data_malicious = normalize_cols(data_malicious)

# Drop last column if dataset is KDD
if args.dataset == 'kdd':
    data_benign = data_benign.drop(data_benign.columns[-1], axis=1)
    data_malicious = data_malicious.drop(data_malicious.columns[-1], axis=1)

#test and train split
train_data, val_data, Y_train, Y_val = train_test_split(data_benign, Y_benign, test_size=0.5, random_state=args.seed)
val_data, test_data, Y_val, Y_test = train_test_split(val_data, Y_val, test_size=0.5, random_state=args.seed)
val_mal_data, test_mal_data, Y_val_mal, Y_test_mal = train_test_split(data_malicious, Y_malicious, test_size=0.5, random_state=args.seed)

val_data = pd.concat([val_data, val_mal_data], ignore_index=True)
test_data = pd.concat([test_data, test_mal_data], ignore_index=True)

# Concatenate Y
Y_val = np.concatenate([Y_val, Y_val_mal])
Y_test = np.concatenate([Y_test, Y_test_mal])


if args.contamination > 0:
    train_data = train_data.reset_index(drop=True)
    val_data = val_data.reset_index(drop=True)
    Y_train = Y_train.flatten()
    Y_val = Y_val.flatten()

    benign_index = np.where(Y_train == 0)[0]
    malicious_index = np.where(Y_val == 1)[0]
    n_samples = int(len(train_data) * args.contamination)

    selected_benign_index = np.random.choice(benign_index, n_samples, replace=False)
    selected_malicious_index = np.random.choice(malicious_index, n_samples, replace=False)

    benign_data = train_data.iloc[selected_benign_index]
    malicious_data = val_data.iloc[selected_malicious_index]

    train_data = train_data.drop(index=selected_benign_index).reset_index(drop=True)
    Y_train = np.delete(Y_train, selected_benign_index)

    val_data = val_data.drop(index=selected_malicious_index).reset_index(drop=True)
    Y_val = np.delete(Y_val, selected_malicious_index)

    train_data = pd.concat([train_data, malicious_data], ignore_index=True)
    Y_train = np.concatenate([Y_train, np.ones(len(malicious_data))])

    val_data = pd.concat([val_data, benign_data], ignore_index=True)
    Y_val = np.concatenate([Y_val, np.zeros(len(benign_data))])


#train_data = train_data.values.astype(np.float32)
print(train_data.shape)

train_data, train_scaler = normalize_cols(train_data)
val_data, scaler = normalize_cols(val_data, train_scaler)
test_data, scaler = normalize_cols(test_data, train_scaler)

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
som = som_train(train_np, x=10, y=10, sigma=1, learning_rate=args.lrsom, iters=10000)

net = SOM_DAGMM(som, dagmm)
optimizer =  optim.Adam(net.parameters(), lr=args.lrdagmm)

OFFSET = 0.2
MAX_TRYS = 5
MAX_ATTEMPT = 5
best_val_score = 0.0
count_since_better_loss = 0

os.makedirs("checkpoints", exist_ok=True)
model_path = os.path.join("checkpoints", f"{args.dataset}_{args.features}_{args.embed}_{args.contamination}_best.pt")

best_val_score = 0.0

for epoch in range(epochs):
    net.train()
    running_loss = 0.0
    error_count = 0

    for batch in train_dataloader:
        data = batch[0]
        optimizer.zero_grad()

        out = net(data)

        L_loss = compression.reconstruction_loss(data)

        # GMM likelihood loss
        G_loss = mix.gmm_loss(out=out, L1=args.l1, L2=args.l2)

        loss = L_loss + G_loss

        if torch.isfinite(loss):
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        else:
            error_count += 1

    net.eval()
    with torch.no_grad():
        out_val = net(val_data)

    threshold = 0
    if args.dataset == 'cic':
        threshold = np.percentile(out_val.cpu().numpy(), 45)
    elif args.dataset == 'kdd':
        threshold = np.percentile(out_val.cpu().numpy(), 25)
    else:
        threshold = 20
    
    pred_val = (out_val.cpu().numpy() > threshold).astype(int)

    acc, prec, rec, f1, _ = get_scores(pred_val, Y_val)

    print(f"Epoch {epoch+1}: Loss={running_loss:.4f}, Errors={error_count}, F1={f1:.4f}")


    if f1 > best_val_score:
        best_val_score = f1
        torch.save(net, model_path)
        print(f"New best model saved to {model_path} with F1={f1:.4f}")
        print(f"Validation - Acc: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
        count_since_better_loss = 0

    else:
        count_since_better_loss += 1
        print(f"Trys since better loss {count_since_better_loss}")
        if count_since_better_loss >= MAX_TRYS:
            break

