"""
=================================================================
Entraînement Self-Supervised (Pretraining)
=================================================================
Ce script entraîne l'encodeur à reconstruire les features masquées.

La loss ne se calcule QUE sur les positions masquées :
    loss = MSE(prediction[mask], original[mask])

Usage :
    python src/train_ssl.py

Sortie :
    models/ssl_encoder.pt (poids de l'encodeur pré-entraîné)
=================================================================
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from dataset import get_dataloaders
from ssl_model import build_model


# ─── Configuration ──────────────────────────────────────────────
EPOCHS = 20
BATCH_SIZE = 512
LEARNING_RATE = 1e-3
MASK_RATIO = 0.15
MODEL_DIR = os.path.join("..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def masked_mse_loss(predictions, targets, mask):
    """
    MSE calculée UNIQUEMENT sur les features masquées.
    
    On ne veut pas que le modèle apprenne à copier les features visibles
    (c'est trivial). On veut qu'il apprenne les RELATIONS entre features
    pour reconstruire celles qu'il ne voit pas.
    """
    masked_pred = predictions * mask
    masked_target = targets * mask
    
    n_masked = mask.sum()
    if n_masked == 0:
        return torch.tensor(0.0, device=predictions.device)
    
    loss = ((masked_pred - masked_target) ** 2).sum() / n_masked
    return loss


def train_one_epoch(model, dataloader, optimizer, device):
    """Entraîne le modèle sur un epoch."""
    model.train()
    total_loss = 0
    n_batches = 0
    
    for x_masked, x_original, mask in dataloader:
        x_masked = x_masked.to(device)
        x_original = x_original.to(device)
        mask = mask.to(device)
        
        # Forward pass
        predictions = model(x_masked, mode="ssl")
        loss = masked_mse_loss(predictions, x_original, mask)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches


def validate(model, dataloader, device):
    """Évalue le modèle sur le set de validation."""
    model.eval()
    total_loss = 0
    n_batches = 0
    
    with torch.no_grad():
        for x_masked, x_original, mask in dataloader:
            x_masked = x_masked.to(device)
            x_original = x_original.to(device)
            mask = mask.to(device)
            
            predictions = model(x_masked, mode="ssl")
            loss = masked_mse_loss(predictions, x_original, mask)
            
            total_loss += loss.item()
            n_batches += 1
    
    return total_loss / n_batches


def train_ssl():
    """Lance le pretraining self-supervised complet."""
    print("=" * 60)
    print("SELF-SUPERVISED PRETRAINING")
    print(f"Device : {DEVICE}")
    print(f"Epochs : {EPOCHS}")
    print(f"Batch  : {BATCH_SIZE}")
    print(f"LR     : {LEARNING_RATE}")
    print(f"Mask   : {MASK_RATIO*100:.0f}%")
    print("=" * 60)
    
    # 1. Charger les données
    loaders = get_dataloaders(
        processed_dir=os.path.join("..", "data", "processed"),
        batch_size=BATCH_SIZE,
        mask_ratio=MASK_RATIO,
        mode="ssl",
        num_workers=0,
    )
    
    n_features = loaders["n_features"]
    
    # 2. Construire le modèle
    model = build_model(input_dim=n_features)
    model = model.to(DEVICE)
    
    # 3. Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5
    )
    
    # 4. Entraînement
    print(f"\nDébut de l'entraînement...\n")
    
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []
    
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        
        train_loss = train_one_epoch(model, loaders["train_loader"], optimizer, DEVICE)
        val_loss = validate(model, loaders["val_loader"], DEVICE)
        
        scheduler.step(val_loss)
        
        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Sauvegarder le meilleur modèle
        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "ssl_best.pt"))
            marker = " ← best"
        
        print(f"  Epoch {epoch:2d}/{EPOCHS} | "
              f"Train Loss: {train_loss:.6f} | "
              f"Val Loss: {val_loss:.6f} | "
              f"LR: {current_lr:.6f} | "
              f"{elapsed:.1f}s{marker}")
    
    # 5. Sauvegarder le modèle final + l'encodeur seul
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "ssl_final.pt"))
    torch.save(model.encoder.state_dict(), os.path.join(MODEL_DIR, "ssl_encoder.pt"))
    
    # 6. Sauvegarder les courbes de loss
    np.savez(
        os.path.join(MODEL_DIR, "ssl_training_history.npz"),
        train_losses=train_losses,
        val_losses=val_losses,
    )
    
    print(f"\nPretraining terminé !")
    print(f"  Best val loss  : {best_val_loss:.6f}")
    print(f"  Modèle sauvé   : {MODEL_DIR}/ssl_encoder.pt")
    print(f"  Historique     : {MODEL_DIR}/ssl_training_history.npz")
    
    return model, train_losses, val_losses


if __name__ == "__main__":
    train_ssl()
