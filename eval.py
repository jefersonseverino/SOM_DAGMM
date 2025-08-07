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

SEED = 42

#read_inputs
def parse_args():
    
    parser = argparse.ArgumentParser(description='Anomaly Detection with unsupervised methods')
    parser.add_argument('--dataset', dest='dataset', help='training dataset', default='kdd', type=str)
    parser.add_argument('--embedding', dest='embed', help='one_hot, label', default='NULL', type=str)
    parser.add_argument('--features', dest='features', help='all, numerical, categorical', default='all', type=str)
    parser.add_argument('--threshold', dest='threshold', help='32', default = 20, type=int)
    parser.add_argument('--contamination', dest='contamination', help='0.1', default=0.0, type=float)
    args = parser.parse_args()
    return args

args = parse_args()
save_path = os.path.join("checkpoints", args.dataset + "_" + args.features + "_" + args.embed + "_" + str(args.contamination) + "_best" + ".pt")
batch_size = 1024
#read data
# get labels from dataset and drop them if available
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
    Y = get_labels(data, args.dataset)

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
    # Get a sample of 500k rows
    data = data.sample(n=500000, random_state=SEED)
    data_benign = data[data['Label'] == 'Benign'].copy()
    data_malicious = data[data['Label'] != 'Benign'].copy()

Y_benign, data_benign = get_labels(data_benign, args.dataset)
Y_malicious, data_malicious = get_labels(data_malicious, args.dataset)


# Remove columns with NA values
# data = fill_na(data)
data_benign = fill_na(data_benign)
data_malicious = fill_na(data_malicious)

# normalize data
# data = normalize_cols(data)
data_benign = normalize_cols(data_benign)
data_malicious = normalize_cols(data_malicious)

if args.dataset == 'kdd':
    data_benign = data_benign.drop(data_benign.columns[-1], axis=1)
    data_malicious = data_malicious.drop(data_malicious.columns[-1], axis=1)

#test and train split
train_data, val_data, Y_train, Y_val = train_test_split(data_benign, Y_benign, test_size=0.5, random_state=SEED)
# Split again for validation and test
val_data, test_data, Y_val, Y_test = train_test_split(val_data, Y_val, test_size=0.5, random_state=SEED)

# Split malicious data only for validation and test
val_mal_data, test_mal_data, Y_val_mal, Y_test_mal = train_test_split(data_malicious, Y_malicious, test_size=0.5, random_state=SEED)

# Concatenate benign and malicious data for validation and test
val_data = pd.concat([val_data, val_mal_data], ignore_index=True)
test_data = pd.concat([test_data, test_mal_data], ignore_index=True)

# Concatenate Y
Y_val = np.concatenate((Y_val, Y_val_mal), axis=0)
Y_test = np.concatenate((Y_test, Y_test_mal), axis=0)

if args.contamination > 0:
    train_data = train_data.reset_index(drop=True)
    val_data = val_data.reset_index(drop=True)
    
    if args.dataset == 'kdd':
        malicious_index = np.where(Y_val == 1)[0]
        benign_index = np.where(Y_train == 0)[0]
    elif args.dataset == 'arrhythmia':
        malicious_index = np.where(Y_val != 1)[0]
        benign_index = np.where(Y_train == 1)[0]

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

net = torch.load(save_path, weights_only=False)
net.eval()
out = net(test_data)
threshold = np.percentile(out, args.threshold)
pred = (out > threshold).numpy().astype(int)

# Precision, Recall, F1
acc, p, r, f, a = get_scores(pred, Y_test)
print("Accuracy:", acc, "Precision:", p, "Recall:", r, "F1 Score:", f, "AUROC:", a)