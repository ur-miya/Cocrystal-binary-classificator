# Cocrystal formation prediction with Graph Neural Networks

This repository contains a binary classifier to predict whether two molecules can form a cocrystal. The model uses a Siamese Graph Neural Network (GNN) to process molecular graphs constructed from SMILES strings. It supports multiple convolutional layers (GCN, GAT, GINE) and can incorporate global molecular descriptors and MACCS fingerprints as additional features.

## Overview

The task is to predict a binary label (0 or 1) indicating whether two given molecules form a cocrystal. Input is provided as SMILES strings for each molecule. The approach:

1. **Molecular Graph Construction** – Convert each SMILES to a graph (atoms = nodes, bonds = edges) with atom and bond features.
2. **Siamese GNN** – Two identical GNN encoders produce embeddings for each molecule.
3. **Interaction & Classification** – Combine embeddings (concatenation or bilinear) and optionally add global features / fingerprints, then pass through a classifier to output a probability.

Supported GNN layers:
- GCN
- GAT
- GINE (with edge features)

## Requirements

- Python 3.8+
- PyTorch
- RDKit
- PyTorch Geometric
- pandas, numpy, scikit-learn, matplotlib, tqdm

## Dataset Format

The training and test CSV files must contain the following columns:
- `SMILES1` : SMILES string of the first molecule.
- `SMILES2` : SMILES string of the second molecule.
- `result`  : Binary label (0/1) – required only for training/validation.
- `id`      : Unique identifier (required for test submissions).

Example:

| id | SMILES1        | SMILES2        | result |
|----|----------------|----------------|--------|
| 0  | CCO            | c1ccccc1       | 1      |
| 1  | CC(=O)O        | CCO            | 0      |

### Training

1. Place your `train.csv` and `test.csv` files in a directory 
   Update the paths in the code accordingly.

2. Run the training script (the provided notebook or a Python script).  
   The script will:
   - Load and split the training data (train/validation).
   - Compute global features and fit a scaler.
   - Create graph datasets.
   - Train the Siamese GNN with configurable hyperparameters.
   - Save the best model as `best_model.pt`.

3. During training, the console shows loss, AUC, and weighted accuracy for each epoch.  
   The weighted accuracy uses custom weights to handle imbalance.

## Model Configuration

You can easily adjust the model architecture and features by changing the following variables in the code:

- `use_global` : whether to include global molecular descriptors (weight, LogP, TPSA, etc.)
- `use_fp` : fingerprint type (`'maccs'`, `'rdkit'`, or `None`)
- `conv_type` : GNN layer (`'GCN'`, `'GAT'`, `'GINE'`)
- `interaction` : how to combine embeddings (`'concat'` or `'bilinear'`)
- `hidden_dim`, `embedding_dim` : dimensions of hidden layers and final embedding
- `batch_size`, `lr`, `epochs`, `patience` : training hyperparameters

The code automatically computes class weights for the loss function if the training set is imbalanced.

## Results

On a typical dataset (80/20 train/validation split), the model achieves:
- Validation AUC: ~0.85-0.90
- Weighted Accuracy: ~0.80-0.85

Performance may vary based on dataset size, imbalance, and feature choices. 
