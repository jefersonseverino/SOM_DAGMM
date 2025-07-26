# SOM_DAGMM
Code for Self Organizing Map assissted Deep Autoencoding gaussian Mixture Model for intrusion detection

            
        
## Data

** Please download the datasets and save in directory named `data`. **

Credit Card - https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud .\
Arrhythmia - https://archive.ics.uci.edu/ml/datasets/arrhythmia. \
KDD - https://www.unb.ca/cic/datasets/nsl.html

## Training

train.py [-h] [--dataset DATASET] [--embedding EMBED] [--features FEATURES] [--embed EMBEDDING] [--batch_size BATCH_SIZE]

` python src/train.py --dataset kdd --embed label_encode --epoch 100 --batch_size 1024`

## Evaluation

eval.py [-h] [--dataset DATASET] [--embedding EMBED] [--features FEATURES] [--threshold THRESHOLD]

` python src/eval.py --dataset kdd --embed label_encode --threshold 20`

## References

1. Research Paper - https://arxiv.org/pdf/2008.12686.pdf
2. Code DAGMM - https://github.com/RomainSabathe/dagmm
3. Code SOM - https://github.com/JustGlowing/minisom
