"""
=================================================================
Fine-tuning — Détection de fraude
=================================================================
Ce script charge l'encodeur pré-entraîné (SSL) et ajoute une
tête de classification pour prédire fraude / non fraude.

L'encodeur est GELÉ : on ne modifie que la tête de classification.
Cela force le modèle à utiliser les représentations apprises
pendant le pretraining.

Usage :
    python src/finetune.py

Sortie :
    models/fraud_detector.pt (modèle final)
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
EPOCHS = 15
BATCH_SIZE = 512
LEARNING_RATE = 5e-4
MODEL_DIR = os.path.join("..", "models")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """Entraîne la tête de classification sur un epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        predictions = model(X_batch, mode="supervised")
        loss = criterion(predictions, y_batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        predicted = (predictions > 0.5).float()
        correct += (predicted == y_batch).sum().item()
        total += y_batch.size(0)
    
    accuracy = correct / total
    avg_loss = total_loss / len(dataloader)
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    """Évalue le modèle sur le set de validation."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            predictions = model(X_batch, mode="supervised")
            loss = criterion(predictions, y_batch)
            
            total_loss += loss.item()
            predicted = (predictions > 0.5).float()
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)
            
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    
    accuracy = correct / total
    avg_loss = total_loss / len(dataloader)
    return avg_loss, accuracy, np.array(all_preds), np.array(all_labels)


def finetune():
    """Lance le fine-tuning pour la détection de fraude."""
    print("=" * 60)
    print("FINE-TUNING — FRAUD DETECTION")
    print(f"Device : {DEVICE}")
    print(f"Epochs : {EPOCHS}")
    print(f"Batch  : {BATCH_SIZE}")
    print(f"LR     : {LEARNING_RATE}")
    print("=" * 60)
    
    # 1. Charger les données en mode supervisé
    loaders = get_dataloaders(
        processed_dir=os.path.join("..", "data", "processed"),
        batch_size=BATCH_SIZE,
        mode="supervised",
        num_workers=0,
    )
    
    n_features = loaders["n_features"]
    
    # 2. Construire le modèle et charger l'encodeur pré-entraîné
    model = build_model(input_dim=n_features)
    
    encoder_path = os.path.join(MODEL_DIR, "ssl_encoder.pt")
    if os.path.exists(encoder_path):
        model.encoder.load_state_dict(torch.load(encoder_path, weights_only=True))
        print(f"\n✓ Encodeur SSL chargé depuis {encoder_path}")
    else:
        print(f"\n⚠ Encodeur SSL non trouvé. Entraînement from scratch.")
    
    # 3. Geler l'encodeur (on entraîne SEULEMENT la tête de classification)
    model.freeze_encoder()
    model = model.to(DEVICE)
    
    # 4. Loss et optimizer
    # pos_weight corrige le déséquilibre : 96.5% non-fraude vs 3.5% fraude
    fraud_ratio = 0.035
    pos_weight = torch.tensor([(1 - fraud_ratio) / fraud_ratio]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # On remplace le Sigmoid dans la tête par rien (BCEWithLogitsLoss le gère)
    # En fait, gardons BCELoss simple puisqu'on a déjà Sigmoid
    criterion = nn.BCELoss()
    
    # Optimizer : seulement les paramètres non gelés
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=LEARNING_RATE)
    
    print(f"  Paramètres entraînables : {sum(p.numel() for p in trainable_params):,}")
    print(f"  Paramètres gelés        : {sum(p.numel() for p in model.parameters() if not p.requires_grad):,}")
    
    # 5. Entraînement
    print(f"\nDébut du fine-tuning...\n")
    
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []
    
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        
        train_loss, train_acc = train_one_epoch(
            model, loaders["train_loader"], optimizer, criterion, DEVICE
        )
        val_loss, val_acc, val_preds, val_labels = validate(
            model, loaders["val_loader"], criterion, DEVICE
        )
        
        elapsed = time.time() - t0
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "fraud_detector_best.pt"))
            marker = " ← best"
        
        print(f"  Epoch {epoch:2d}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
              f"{elapsed:.1f}s{marker}")
    
    # 6. Sauvegarder
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "fraud_detector_final.pt"))
    np.savez(
        os.path.join(MODEL_DIR, "finetune_history.npz"),
        train_losses=train_losses,
        val_losses=val_losses,
        val_preds=val_preds,
        val_labels=val_labels,
    )
    
    print(f"\nFine-tuning terminé !")
    print(f"  Best val loss : {best_val_loss:.4f}")
    print(f"  Modèle sauvé  : {MODEL_DIR}/fraud_detector_best.pt")
    
    return model


if __name__ == "__main__":
    finetune()
