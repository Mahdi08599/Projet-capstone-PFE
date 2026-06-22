"""
=================================================================
Fine-tuning V2 — Version améliorée
=================================================================
Améliorations par rapport à la V1 :
  1. L'encodeur est DÉGELÉ avec un learning rate plus faible
  2. La loss utilise pos_weight pour compenser le déséquilibre
  3. On cherche le seuil optimal sur la validation

Usage :
    python src/finetune_v2.py
=================================================================
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score

from dataset import get_dataloaders
from ssl_model import SSLFraudModel, TransactionEncoder, FraudClassificationHead


# ─── Configuration ──────────────────────────────────────────────
EPOCHS = 20
BATCH_SIZE = 512
LR_ENCODER = 1e-4        # learning rate faible pour l'encodeur (ne pas casser ce qu'il a appris)
LR_HEAD = 5e-4            # learning rate normal pour la tête de classification
MODEL_DIR = os.path.join("..", "models")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    n = 0

    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(X_batch, mode="supervised")
        loss = criterion(logits, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y_batch.size(0)
        n += y_batch.size(0)

    return total_loss / n


def get_predictions(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            preds = model(X_batch, mode="supervised")
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.numpy())

    return np.array(all_preds), np.array(all_labels)


def find_best_threshold(y_true, y_proba):
    """Teste plusieurs seuils et garde celui qui maximise le F1-score."""
    best_f1 = 0
    best_t = 0.5

    for t in np.arange(0.05, 0.95, 0.05):
        y_pred = (y_proba >= t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    return best_t, best_f1


def finetune_v2():
    print("=" * 60)
    print("FINE-TUNING V2 — IMPROVED")
    print(f"Device : {DEVICE}")
    print("=" * 60)

    # 1. Charger les données
    loaders = get_dataloaders(
        processed_dir=os.path.join("..", "data", "processed"),
        batch_size=BATCH_SIZE,
        mode="supervised",
        num_workers=0,
    )
    n_features = loaders["n_features"]

    # 2. Construire le modèle et charger l'encodeur SSL
    model = SSLFraudModel(input_dim=n_features)

    encoder_path = os.path.join(MODEL_DIR, "ssl_encoder.pt")
    if os.path.exists(encoder_path):
        model.encoder.load_state_dict(torch.load(encoder_path, weights_only=True))
        print(f"✓ Encodeur SSL chargé")
    else:
        print(f"⚠ Encodeur SSL non trouvé")

    model = model.to(DEVICE)

    # 3. Deux learning rates différents
    optimizer = torch.optim.Adam([
        {"params": model.encoder.parameters(), "lr": LR_ENCODER},
        {"params": model.classification_head.parameters(), "lr": LR_HEAD},
    ])

    # 4. Loss avec pos_weight pour compenser le déséquilibre
    #    ~96.5% non-fraude vs ~3.5% fraude → ratio ~27.5
    pos_weight = torch.tensor([27.0]).to(DEVICE)

    # On a besoin de logits (pas de sigmoid) pour BCEWithLogitsLoss
    # Donc on bypass le sigmoid dans le forward
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Remplacer le forward pour sortir des logits
    original_classif = model.classification_head

    class LogitHead(nn.Module):
        def __init__(self, original):
            super().__init__()
            self.layers = nn.Sequential(
                original.network[0],  # Linear 64→32
                original.network[1],  # ReLU
                original.network[2],  # Dropout
                original.network[3],  # Linear 32→1
                # PAS de Sigmoid
            )
        def forward(self, z):
            return self.layers(z).squeeze(-1)

    model.classification_head = LogitHead(original_classif).to(DEVICE)

    print(f"LR encoder : {LR_ENCODER}")
    print(f"LR head    : {LR_HEAD}")
    print(f"pos_weight : {pos_weight.item()}")

    # 5. Entraînement
    print(f"\nDébut du fine-tuning...\n")

    best_val_f1 = 0
    train_losses = []
    val_losses = []

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        train_loss = train_one_epoch(
            model, loaders["train_loader"], optimizer, criterion, DEVICE
        )

        # Validation
        y_proba_raw, y_true = get_predictions(model, loaders["val_loader"], DEVICE)
        y_proba = 1 / (1 + np.exp(-y_proba_raw))  # sigmoid sur les logits

        best_t, best_f1 = find_best_threshold(y_true, y_proba)
        y_pred = (y_proba >= best_t).astype(int)
        recall = recall_score(y_true, y_pred, zero_division=0)
        precision = precision_score(y_true, y_pred, zero_division=0)

        elapsed = time.time() - t0
        train_losses.append(train_loss)

        marker = ""
        if best_f1 > best_val_f1:
            best_val_f1 = best_f1
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "fraud_v2_best.pt"))
            best_preds = y_proba
            best_labels = y_true
            best_threshold = best_t
            marker = " ← best"

        print(f"  Epoch {epoch:2d}/{EPOCHS} | "
              f"Loss: {train_loss:.4f} | "
              f"F1: {best_f1:.4f} | "
              f"Prec: {precision:.4f} | "
              f"Rec: {recall:.4f} | "
              f"Seuil: {best_t:.2f} | "
              f"{elapsed:.1f}s{marker}")

    # 6. Sauvegarder
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "fraud_v2_final.pt"))
    np.savez(
        os.path.join(MODEL_DIR, "finetune_v2_history.npz"),
        train_losses=train_losses,
        val_preds=best_preds,
        val_labels=best_labels,
        best_threshold=best_threshold,
    )

    print(f"\nFine-tuning V2 terminé !")
    print(f"  Best F1       : {best_val_f1:.4f}")
    print(f"  Best seuil    : {best_threshold:.2f}")
    print(f"  Modèle sauvé  : {MODEL_DIR}/fraud_v2_best.pt")


if __name__ == "__main__":
    finetune_v2()
