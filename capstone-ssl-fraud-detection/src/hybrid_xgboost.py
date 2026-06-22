"""
=================================================================
Approche Hybride : SSL Embeddings + XGBoost
=================================================================
Au lieu d'utiliser un petit MLP pour classifier, on :
  1. Passe toutes les transactions dans l'encodeur SSL
  2. Récupère les représentations latentes (64 dimensions)
  3. Entraîne XGBoost sur ces représentations

Pourquoi c'est mieux :
  - XGBoost est le meilleur classifieur pour les données tabulaires
  - Les embeddings SSL capturent la structure des transactions
  - On combine le meilleur des deux mondes

Usage :
    python src/hybrid_xgboost.py
=================================================================
"""

import os
import time
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score,
)
from sklearn.ensemble import GradientBoostingClassifier

from ssl_model import SSLFraudModel

MODEL_DIR = os.path.join("..", "models")
DATA_DIR = os.path.join("..", "data", "processed")
FIGURES_DIR = os.path.join("..", "reports", "figures")
RESULTS_DIR = os.path.join("..", "reports", "results")
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extract_embeddings(model, X, batch_size=1024):
    """Passe les données dans l'encodeur et récupère les embeddings."""
    model.eval()
    embeddings = []
    
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.FloatTensor(X[i:i+batch_size]).to(DEVICE)
            z = model.encoder(batch)
            embeddings.append(z.cpu().numpy())
    
    return np.vstack(embeddings)


def run():
    print("=" * 60)
    print("APPROCHE HYBRIDE : SSL + XGBoost")
    print("=" * 60)
    
    # 1. Charger les données
    print("\n1. Chargement des données...")
    train_df = pd.read_parquet(f"{DATA_DIR}/train_clean.parquet")
    val_df = pd.read_parquet(f"{DATA_DIR}/val_clean.parquet")
    
    y_train = train_df["isFraud"].values
    X_train = train_df.drop(columns=["isFraud"]).values
    y_val = val_df["isFraud"].values
    X_val = val_df.drop(columns=["isFraud"]).values
    
    n_features = X_train.shape[1]
    print(f"   Train: {len(X_train):,} | Val: {len(X_val):,} | Features: {n_features}")
    
    # 2. Charger l'encodeur SSL pré-entraîné
    print("\n2. Chargement de l'encodeur SSL...")
    model = SSLFraudModel(input_dim=n_features)
    encoder_path = os.path.join(MODEL_DIR, "ssl_encoder.pt")
    model.encoder.load_state_dict(torch.load(encoder_path, weights_only=True))
    model = model.to(DEVICE)
    print(f"   ✓ Encodeur chargé")
    
    # 3. Extraire les embeddings SSL
    print("\n3. Extraction des embeddings SSL (64 dimensions)...")
    t0 = time.time()
    Z_train = extract_embeddings(model, X_train)
    Z_val = extract_embeddings(model, X_val)
    print(f"   Train embeddings: {Z_train.shape}")
    print(f"   Val embeddings:   {Z_val.shape}")
    print(f"   ✓ Extraction en {time.time()-t0:.1f}s")
    
    # 4. Option A : XGBoost sur embeddings SSL seuls
    print("\n4. Entraînement XGBoost sur embeddings SSL...")
    t0 = time.time()
    
    fraud_ratio = y_train.sum() / len(y_train)
    scale_pos = (1 - fraud_ratio) / fraud_ratio
    
    xgb_ssl = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        min_samples_leaf=50,
        random_state=42,
    )
    xgb_ssl.fit(Z_train, y_train)
    
    y_proba_ssl = xgb_ssl.predict_proba(Z_val)[:, 1]
    print(f"   ✓ Entraîné en {time.time()-t0:.1f}s")
    
    # 5. Option B : XGBoost sur embeddings + features originales
    print("\n5. Entraînement XGBoost sur embeddings + features originales...")
    t0 = time.time()
    
    X_train_combined = np.hstack([X_train, Z_train])
    X_val_combined = np.hstack([X_val, Z_val])
    print(f"   Features combinées: {X_train_combined.shape[1]} ({n_features} originales + 64 SSL)")
    
    xgb_combined = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        min_samples_leaf=50,
        random_state=42,
    )
    xgb_combined.fit(X_train_combined, y_train)
    
    y_proba_combined = xgb_combined.predict_proba(X_val_combined)[:, 1]
    print(f"   ✓ Entraîné en {time.time()-t0:.1f}s")
    
    # 6. Option C : Baseline — XGBoost sur features originales (SANS SSL)
    print("\n6. Baseline : XGBoost sur features originales (sans SSL)...")
    t0 = time.time()
    
    xgb_baseline = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        min_samples_leaf=50,
        random_state=42,
    )
    xgb_baseline.fit(X_train, y_train)
    
    y_proba_baseline = xgb_baseline.predict_proba(X_val)[:, 1]
    print(f"   ✓ Entraîné en {time.time()-t0:.1f}s")
    
    # 7. Résultats comparés
    print("\n" + "=" * 70)
    print("                    COMPARAISON DES APPROCHES")
    print("=" * 70)
    
    results = {}
    for name, y_proba in [
        ("Baseline (XGBoost seul)", y_proba_baseline),
        ("SSL embeddings + XGBoost", y_proba_ssl),
        ("SSL + originales + XGBoost", y_proba_combined),
    ]:
        # Trouver le meilleur seuil
        best_f1, best_t = 0, 0.5
        for t in np.arange(0.05, 0.95, 0.05):
            f1 = f1_score(y_val, (y_proba >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        
        y_pred = (y_proba >= best_t).astype(int)
        auc = roc_auc_score(y_val, y_proba)
        prec = precision_score(y_val, y_pred, zero_division=0)
        rec = recall_score(y_val, y_pred, zero_division=0)
        
        results[name] = {
            "auc": auc, "precision": prec, "recall": rec,
            "f1": best_f1, "threshold": best_t, "y_proba": y_proba
        }
        
        print(f"\n  {name}:")
        print(f"    AUC-ROC   = {auc:.4f}")
        print(f"    Precision = {prec:.4f}")
        print(f"    Recall    = {rec:.4f}")
        print(f"    F1-score  = {best_f1:.4f}")
        print(f"    Seuil     = {best_t:.2f}")
    
    # 8. Graphiques
    print("\n\n8. Génération des graphiques...")
    
    # ── ROC comparée ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = ["#888888", "#2196F3", "#E53935"]
    for (name, r), color in zip(results.items(), colors):
        fpr, tpr, _ = roc_curve(y_val, r["y_proba"])
        ax.plot(fpr, tpr, linewidth=2.5, label=f"{name} (AUC={r['auc']:.4f})", color=color)
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=0.8)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — Comparison", fontsize=14)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "roc_comparison.png"), dpi=150)
    plt.close()
    print("   ✓ roc_comparison.png")
    
    # ── Barplot comparatif ──────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(14, 5))
    metrics_names = ["auc", "precision", "recall", "f1"]
    titles = ["AUC-ROC", "Precision", "Recall", "F1-score"]
    short_names = ["Baseline", "SSL only", "SSL + orig."]
    
    for ax, metric, title in zip(axes, metrics_names, titles):
        values = [results[n][metric] for n in results]
        bars = ax.bar(short_names, values, color=colors, alpha=0.85)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"{val:.3f}", ha="center", fontsize=9)
    
    fig.suptitle("Comparison: Baseline vs SSL approaches", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "metrics_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("   ✓ metrics_comparison.png")
    
    # ── Confusion matrix du meilleur modèle ─────────────────
    best_name = max(results, key=lambda k: results[k]["f1"])
    best_r = results[best_name]
    y_pred_best = (best_r["y_proba"] >= best_r["threshold"]).astype(int)
    cm = confusion_matrix(y_val, y_pred_best)
    
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues",
                xticklabels=["Légitime", "Fraude"],
                yticklabels=["Légitime", "Fraude"], ax=ax)
    ax.set_xlabel("Prédit", fontsize=12)
    ax.set_ylabel("Réel", fontsize=12)
    ax.set_title(f"Confusion Matrix — {best_name}", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "confusion_best.png"), dpi=150)
    plt.close()
    print("   ✓ confusion_best.png")
    
    # 9. Sauvegarder le rapport
    with open(os.path.join(RESULTS_DIR, "comparison_results.txt"), "w") as f:
        f.write("COMPARISON RESULTS\n")
        f.write("=" * 60 + "\n\n")
        for name, r in results.items():
            f.write(f"{name}:\n")
            f.write(f"  AUC={r['auc']:.4f}  Prec={r['precision']:.4f}  "
                    f"Rec={r['recall']:.4f}  F1={r['f1']:.4f}  Seuil={r['threshold']:.2f}\n\n")
    print("   ✓ comparison_results.txt")
    
    print("\n" + "=" * 60)
    print(f"MEILLEUR MODÈLE : {best_name}")
    print(f"  F1 = {best_r['f1']:.4f} | AUC = {best_r['auc']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    run()
