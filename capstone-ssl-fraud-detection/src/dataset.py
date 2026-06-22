"""
=================================================================
Dataset & DataLoader pour le Self-Supervised Learning
=================================================================
Ce module fournit :
    - TransactionDataset : dataset PyTorch pour les transactions
    - MaskedTransactionDataset : version avec masquage aléatoire
      de features pour la tâche SSL (reconstruction)
    - get_dataloaders() : fonction utilitaire pour créer les loaders

La tâche self-supervised :
    On masque aléatoirement ~15% des features d'une transaction,
    et le modèle doit reconstruire les valeurs originales.
=================================================================
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class TransactionDataset(Dataset):
    """
    Dataset PyTorch standard pour les transactions.
    Utilisé pour le fine-tuning supervisé (classification fraude).
    """
    def __init__(self, X: np.ndarray, y: np.ndarray = None):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y) if y is not None else None
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


class MaskedTransactionDataset(Dataset):
    """
    Dataset avec masquage aléatoire pour le Self-Supervised Learning.
    
    À chaque appel de __getitem__ :
        1. On prend une transaction X_original
        2. On choisit aléatoirement mask_ratio % des features
        3. On les remplace par 0 (masquage)
        4. On retourne (X_masked, X_original, mask)
    
    Le modèle doit prédire X_original à partir de X_masked.
    La loss ne se calcule QUE sur les positions masquées (mask == 1).
    
    Args:
        X: features normalisées (np.ndarray)
        mask_ratio: proportion de features à masquer (default 0.15)
    """
    def __init__(self, X: np.ndarray, mask_ratio: float = 0.15):
        self.X = torch.FloatTensor(X)
        self.mask_ratio = mask_ratio
        self.n_features = X.shape[1]
        self.n_mask = max(1, int(self.n_features * mask_ratio))
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x_original = self.X[idx].clone()
        
        # Créer le masque : 1 = masqué, 0 = visible
        mask = torch.zeros(self.n_features)
        mask_indices = torch.randperm(self.n_features)[:self.n_mask]
        mask[mask_indices] = 1.0
        
        # Appliquer le masque (remplacer par 0)
        x_masked = x_original.clone()
        x_masked[mask.bool()] = 0.0
        
        return x_masked, x_original, mask


def get_dataloaders(
    processed_dir: str = "data/processed",
    batch_size: int = 512,
    mask_ratio: float = 0.15,
    num_workers: int = 2,
    mode: str = "ssl",
) -> dict:
    """
    Crée les DataLoaders pour l'entraînement.
    
    Args:
        processed_dir: chemin vers les données nettoyées
        batch_size: taille du batch
        mask_ratio: ratio de masquage (mode SSL uniquement)
        num_workers: workers pour le chargement parallèle
        mode: "ssl" (pretraining) ou "supervised" (fine-tuning fraude)
    
    Returns:
        dict avec "train_loader" et "val_loader"
    """
    # Charger les données
    train_df = pd.read_parquet(f"{processed_dir}/train_clean.parquet")
    val_df = pd.read_parquet(f"{processed_dir}/val_clean.parquet")
    
    # Séparer features et labels
    y_train = train_df["isFraud"].values
    X_train = train_df.drop(columns=["isFraud"]).values
    
    y_val = val_df["isFraud"].values
    X_val = val_df.drop(columns=["isFraud"]).values
    
    if mode == "ssl":
        # Mode Self-Supervised : masquage de features
        train_dataset = MaskedTransactionDataset(X_train, mask_ratio=mask_ratio)
        val_dataset = MaskedTransactionDataset(X_val, mask_ratio=mask_ratio)
    elif mode == "supervised":
        # Mode supervisé : classification fraude
        train_dataset = TransactionDataset(X_train, y_train)
        val_dataset = TransactionDataset(X_val, y_val)
    else:
        raise ValueError(f"Mode inconnu : {mode}. Utiliser 'ssl' ou 'supervised'.")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    print(f"DataLoaders créés (mode={mode}) :")
    print(f"  Train : {len(train_dataset):,} samples, {len(train_loader)} batches")
    print(f"  Val   : {len(val_dataset):,} samples, {len(val_loader)} batches")
    print(f"  Features : {X_train.shape[1]}")
    if mode == "ssl":
        print(f"  Mask ratio : {mask_ratio} ({int(X_train.shape[1]*mask_ratio)} features masquées/sample)")
    
    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "n_features": X_train.shape[1],
    }
