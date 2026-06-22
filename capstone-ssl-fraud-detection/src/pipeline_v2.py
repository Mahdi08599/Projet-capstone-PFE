"""
=================================================================
Pipeline V2 complet : SSL + Limited Labels
=================================================================
Ce script fait tout d'un coup sur les données V2 :
  1. Entraîner l'encodeur SSL sur 743 features
  2. Comparer Baseline vs SSL avec labels limités

Usage :
    python src/pipeline_v2.py
=================================================================
"""

import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score

# ─── Config ─────────────────────────────────────────────────────
DATA_DIR = os.path.join("..", "data", "processed")
MODEL_DIR = os.path.join("..", "models")
FIGURES_DIR = os.path.join("..", "reports", "figures")
RESULTS_DIR = os.path.join("..", "reports", "results")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
SSL_EPOCHS = 20
SSL_BATCH = 512
SSL_LR = 1e-3
MASK_RATIO = 0.15

LABEL_FRACTIONS = [0.01, 0.02, 0.05, 0.10, 0.25, 0.50, 1.0]


# ─── Modèle SSL (adapté au nombre de features) ─────────────────
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, latent_dim=96):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim=96, hidden_dim=256, output_dim=743):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )
    def forward(self, z):
        return self.net(z)


class SSLModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, latent_dim=96):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


# ─── Fonctions utilitaires ──────────────────────────────────────
def masked_mse_loss(pred, target, mask):
    n = mask.sum()
    if n == 0:
        return torch.tensor(0.0, device=pred.device)
    return ((pred * mask - target * mask) ** 2).sum() / n


def extract_embeddings(encoder, X, batch_size=1024):
    encoder.eval()
    embs = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.FloatTensor(X[i:i+batch_size]).to(DEVICE)
            embs.append(encoder(batch).cpu().numpy())
    return np.vstack(embs)


def subsample(X, y, frac, seed=SEED):
    rng = np.random.RandomState(seed)
    idx_f = np.where(y == 1)[0]
    idx_l = np.where(y == 0)[0]
    n_f = max(int(len(idx_f) * frac), 10)
    n_l = max(int(len(idx_l) * frac), 50)
    sel = np.concatenate([
        rng.choice(idx_f, n_f, replace=False),
        rng.choice(idx_l, n_l, replace=False),
    ])
    return X[sel], y[sel]


def eval_model(X_tr, y_tr, X_val, y_val):
    clf = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        subsample=0.8, min_samples_leaf=30, random_state=SEED,
    )
    clf.fit(X_tr, y_tr)
    y_p = clf.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_p)

    best_f1, best_t = 0, 0.5
    for t in np.arange(0.05, 0.95, 0.05):
        f1 = f1_score(y_val, (y_p >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    y_pred = (y_p >= best_t).astype(int)
    rec = recall_score(y_val, y_pred, zero_division=0)
    prec = precision_score(y_val, y_pred, zero_division=0)
    return {"auc": auc, "f1": best_f1, "recall": rec, "precision": prec}


# =================================================================
# MAIN
# =================================================================
def run():
    print("=" * 65)
    print("  PIPELINE V2 : SSL + LIMITED LABELS (données améliorées)")
    print("=" * 65)

    # ── 1. Charger les données V2 ────────────────────────────
    print("\n1. Chargement des données V2...")
    train_df = pd.read_parquet(f"{DATA_DIR}/train_clean_v2.parquet")
    val_df = pd.read_parquet(f"{DATA_DIR}/val_clean_v2.parquet")

    y_train = train_df["isFraud"].values
    X_train = train_df.drop(columns=["isFraud"]).values.astype(np.float32)
    y_val = val_df["isFraud"].values
    X_val = val_df.drop(columns=["isFraud"]).values.astype(np.float32)

    n_feat = X_train.shape[1]
    print(f"   Train: {len(X_train):,} | Val: {len(X_val):,} | Features: {n_feat}")
    print(f"   (V1 avait 224 features, V2 en a {n_feat})")

    # ── 2. Entraîner le SSL V2 ───────────────────────────────
    print(f"\n2. Entraînement SSL V2 ({SSL_EPOCHS} epochs, {n_feat} features)...")

    model = SSLModel(input_dim=n_feat).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=SSL_LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    n_mask = max(1, int(n_feat * MASK_RATIO))
    X_tensor = torch.FloatTensor(X_train)

    dataset = torch.utils.data.TensorDataset(X_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=SSL_BATCH, shuffle=True, num_workers=0)

    train_losses = []
    for epoch in range(1, SSL_EPOCHS + 1):
        model.train()
        epoch_loss = 0
        n_batches = 0

        for (batch,) in loader:
            batch = batch.to(DEVICE)
            mask = torch.zeros_like(batch)
            for i in range(len(batch)):
                idx = torch.randperm(n_feat)[:n_mask]
                mask[i, idx] = 1.0

            x_masked = batch.clone()
            x_masked[mask.bool()] = 0.0

            pred = model(x_masked)
            loss = masked_mse_loss(pred, batch, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        train_losses.append(avg_loss)
        scheduler.step(avg_loss)
        lr = optimizer.param_groups[0]["lr"]

        print(f"   Epoch {epoch:2d}/{SSL_EPOCHS} | Loss: {avg_loss:.6f} | LR: {lr:.6f}")

    # Sauvegarder
    torch.save(model.encoder.state_dict(), os.path.join(MODEL_DIR, "ssl_encoder_v2.pt"))
    np.savez(os.path.join(MODEL_DIR, "ssl_v2_history.npz"), train_losses=train_losses)
    print(f"   ✓ Encodeur V2 sauvé")

    # ── 3. Extraire les embeddings V2 ────────────────────────
    print(f"\n3. Extraction des embeddings V2...")
    Z_train = extract_embeddings(model.encoder, X_train)
    Z_val = extract_embeddings(model.encoder, X_val)
    X_train_combined = np.hstack([X_train, Z_train])
    X_val_combined = np.hstack([X_val, Z_val])
    print(f"   Features combinées : {X_train_combined.shape[1]} ({n_feat} + {Z_train.shape[1]} SSL)")

    # ── 4. Expérience limited labels V2 ──────────────────────
    print(f"\n4. Expérience limited labels V2...\n")

    results_base = []
    results_ssl = []

    for frac in LABEL_FRACTIONS:
        n_labels = max(int(len(y_train) * frac), 100)
        print(f"   [{frac*100:5.1f}%] {n_labels:,} samples...", end="", flush=True)
        t0 = time.time()

        X_sub, y_sub = subsample(X_train, y_train, frac)
        X_sub_c, _ = subsample(X_train_combined, y_train, frac)

        r_b = eval_model(X_sub, y_sub, X_val, y_val)
        r_s = eval_model(X_sub_c, y_sub, X_val_combined, y_val)

        results_base.append(r_b)
        results_ssl.append(r_s)

        g_auc = r_s["auc"] - r_b["auc"]
        g_f1 = r_s["f1"] - r_b["f1"]
        print(f"  Base AUC={r_b['auc']:.4f} F1={r_b['f1']:.4f} | "
              f"SSL AUC={r_s['auc']:.4f} F1={r_s['f1']:.4f} | "
              f"Gain: {'+' if g_auc>=0 else ''}{g_auc:.4f} AUC, "
              f"{'+' if g_f1>=0 else ''}{g_f1:.4f} F1 | "
              f"{time.time()-t0:.0f}s")

    # ── 5. Résultats ─────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"{'Labels':>8} | {'Baseline AUC':>13} {'F1':>7} | {'SSL AUC':>10} {'F1':>7} | {'Gain AUC':>9} {'Gain F1':>8}")
    print(f"{'-'*70}")
    for i, frac in enumerate(LABEL_FRACTIONS):
        b, s = results_base[i], results_ssl[i]
        ga, gf = s["auc"]-b["auc"], s["f1"]-b["f1"]
        print(f"{frac*100:7.1f}% | {b['auc']:>12.4f} {b['f1']:>7.4f} | "
              f"{s['auc']:>9.4f} {s['f1']:>7.4f} | "
              f"{'+' if ga>=0 else ''}{ga:>8.4f} {'+' if gf>=0 else ''}{gf:>7.4f}")
    print(f"{'='*70}")

    # ── 6. Graphiques comparatifs V1 vs V2 ───────────────────
    print("\n6. Graphiques...")
    fracs_pct = [f*100 for f in LABEL_FRACTIONS]

    # Charger résultats V1 si disponibles
    v1_path = os.path.join(RESULTS_DIR, "limited_labels_results.txt")
    has_v1 = os.path.exists(v1_path)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].plot(fracs_pct, [r["auc"] for r in results_base], "o--",
                 label="Baseline V2", linewidth=2, color="#888888")
    axes[0].plot(fracs_pct, [r["auc"] for r in results_ssl], "s-",
                 label="SSL V2 + XGBoost", linewidth=2.5, color="#E53935")
    axes[0].set_xlabel("% de labels", fontsize=12)
    axes[0].set_ylabel("AUC-ROC", fontsize=12)
    axes[0].set_title("AUC-ROC (Preprocessing V2)", fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xscale("log")
    axes[0].set_xticks(fracs_pct)
    axes[0].set_xticklabels([f"{f:.0f}%" if f>=1 else f"{f}%" for f in fracs_pct])

    axes[1].plot(fracs_pct, [r["f1"] for r in results_base], "o--",
                 label="Baseline V2", linewidth=2, color="#888888")
    axes[1].plot(fracs_pct, [r["f1"] for r in results_ssl], "s-",
                 label="SSL V2 + XGBoost", linewidth=2.5, color="#E53935")
    axes[1].set_xlabel("% de labels", fontsize=12)
    axes[1].set_ylabel("F1-score", fontsize=12)
    axes[1].set_title("F1-score (Preprocessing V2)", fontsize=14)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xscale("log")
    axes[1].set_xticks(fracs_pct)
    axes[1].set_xticklabels([f"{f:.0f}%" if f>=1 else f"{f}%" for f in fracs_pct])

    fig.suptitle("Impact du SSL avec preprocessing ameliore (V2)", fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "limited_labels_v2.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("   ✓ limited_labels_v2.png")

    # Gain chart
    gains_auc = [results_ssl[i]["auc"]-results_base[i]["auc"] for i in range(len(LABEL_FRACTIONS))]
    gains_f1 = [results_ssl[i]["f1"]-results_base[i]["f1"] for i in range(len(LABEL_FRACTIONS))]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(LABEL_FRACTIONS))
    w = 0.35
    ax.bar(x-w/2, gains_auc, w, label="Gain AUC", color="#2196F3", alpha=0.8)
    ax.bar(x+w/2, gains_f1, w, label="Gain F1", color="#E53935", alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("% de labels", fontsize=12)
    ax.set_ylabel("Gain SSL", fontsize=12)
    ax.set_title("Apport du SSL (Preprocessing V2)", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{f*100:.0f}%" for f in LABEL_FRACTIONS])
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "ssl_gain_v2.png"), dpi=150)
    plt.close()
    print("   ✓ ssl_gain_v2.png")

    # Save results
    with open(os.path.join(RESULTS_DIR, "limited_labels_v2_results.txt"), "w", encoding="utf-8") as f:
        f.write("LIMITED LABELS EXPERIMENT — V2 PREPROCESSING\n")
        f.write("=" * 60 + "\n")
        f.write(f"Features: {n_feat} (V1: 224)\n")
        f.write(f"SSL latent dim: 96 (V1: 64)\n\n")
        for i, frac in enumerate(LABEL_FRACTIONS):
            b, s = results_base[i], results_ssl[i]
            f.write(f"{frac*100:.1f}% labels:\n")
            f.write(f"  Baseline: AUC={b['auc']:.4f} F1={b['f1']:.4f} Rec={b['recall']:.4f}\n")
            f.write(f"  SSL:      AUC={s['auc']:.4f} F1={s['f1']:.4f} Rec={s['recall']:.4f}\n")
            f.write(f"  Gain:     AUC={s['auc']-b['auc']:+.4f} F1={s['f1']-b['f1']:+.4f}\n\n")
    print("   ✓ limited_labels_v2_results.txt")

    print(f"\n{'='*65}")
    print("PIPELINE V2 TERMINE")
    print(f"{'='*65}")


if __name__ == "__main__":
    run()
