"""
=================================================================
Évaluation — Métriques et visualisations
=================================================================
Ce script génère toutes les métriques et graphiques nécessaires
pour le mémoire et la soutenance :

    - AUC-ROC + courbe ROC
    - Precision, Recall, F1-score
    - Matrice de confusion
    - Courbes de loss (pretraining + fine-tuning)

Usage :
    python src/evaluate.py

Sortie :
    reports/figures/*.png (graphiques)
    reports/results/metrics.txt (métriques)
=================================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score,
)

MODEL_DIR = os.path.join("..", "models")
FIGURES_DIR = os.path.join("..", "reports", "figures")
RESULTS_DIR = os.path.join("..", "reports", "results")
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")


def plot_ssl_training_curves():
    """Trace la courbe de loss du pretraining SSL."""
    path = os.path.join(MODEL_DIR, "ssl_training_history.npz")
    if not os.path.exists(path):
        print("  ⚠ Pas d'historique SSL trouvé")
        return
    
    data = np.load(path)
    train_losses = data["train_losses"]
    val_losses = data["val_losses"]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, "o-", label="Train loss", linewidth=2)
    ax.plot(epochs, val_losses, "s-", label="Validation loss", linewidth=2)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Masked MSE Loss", fontsize=12)
    ax.set_title("Self-Supervised Pretraining — Reconstruction Loss", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "ssl_training_curve.png"), dpi=150)
    plt.close()
    print("  ✓ ssl_training_curve.png")


def plot_finetune_curves():
    """Trace les courbes de loss du fine-tuning."""
    path = os.path.join(MODEL_DIR, "finetune_history.npz")
    if not os.path.exists(path):
        print("  ⚠ Pas d'historique fine-tuning trouvé")
        return
    
    data = np.load(path)
    train_losses = data["train_losses"]
    val_losses = data["val_losses"]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, "o-", label="Train loss", linewidth=2)
    ax.plot(epochs, val_losses, "s-", label="Validation loss", linewidth=2)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("BCE Loss", fontsize=12)
    ax.set_title("Fine-tuning — Fraud Detection Loss", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "finetune_training_curve.png"), dpi=150)
    plt.close()
    print("  ✓ finetune_training_curve.png")


def compute_metrics():
    """Calcule toutes les métriques et génère les graphiques."""
    path = os.path.join(MODEL_DIR, "finetune_history.npz")
    if not os.path.exists(path):
        print("  ⚠ Pas de prédictions trouvées")
        return
    
    data = np.load(path)
    y_true = data["val_labels"]
    y_proba = data["val_preds"]
    y_pred = (y_proba > 0.5).astype(int)
    
    # ── Métriques ───────────────────────────────────────────────
    auc = roc_auc_score(y_true, y_proba)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    ap = average_precision_score(y_true, y_proba)
    
    print(f"\n  ┌─────────────────────────────────────┐")
    print(f"  │         RÉSULTATS FINAUX             │")
    print(f"  ├─────────────────────────────────────┤")
    print(f"  │  AUC-ROC          : {auc:.4f}            │")
    print(f"  │  Avg Precision    : {ap:.4f}            │")
    print(f"  │  Precision        : {precision:.4f}            │")
    print(f"  │  Recall           : {recall:.4f}            │")
    print(f"  │  F1-score         : {f1:.4f}            │")
    print(f"  └─────────────────────────────────────┘")
    
    # Sauvegarder les métriques
    report = classification_report(y_true, y_pred, target_names=["Légitimes", "Fraudes"])
    metrics_path = os.path.join(RESULTS_DIR, "metrics.txt")
    with open(metrics_path, "w") as f:
        f.write("=" * 50 + "\n")
        f.write("FRAUD DETECTION RESULTS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"AUC-ROC           : {auc:.4f}\n")
        f.write(f"Average Precision : {ap:.4f}\n")
        f.write(f"Precision         : {precision:.4f}\n")
        f.write(f"Recall            : {recall:.4f}\n")
        f.write(f"F1-score          : {f1:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    print(f"  ✓ metrics.txt")
    
    # ── Courbe ROC ──────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(fpr, tpr, linewidth=2.5, label=f"SSL + Fine-tuning (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, label="Random (AUC = 0.5)")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=12)
    ax.set_title("ROC Curve — Fraud Detection", fontsize=14)
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "roc_curve.png"), dpi=150)
    plt.close()
    print("  ✓ roc_curve.png")
    
    # ── Courbe Precision-Recall ─────────────────────────────────
    prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_proba)
    
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(rec_curve, prec_curve, linewidth=2.5, label=f"AP = {ap:.4f}")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve", fontsize=14)
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "precision_recall_curve.png"), dpi=150)
    plt.close()
    print("  ✓ precision_recall_curve.png")
    
    # ── Matrice de confusion ────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt=",d", cmap="Blues",
        xticklabels=["Légitime", "Fraude"],
        yticklabels=["Légitime", "Fraude"],
        ax=ax,
    )
    ax.set_xlabel("Prédit", fontsize=12)
    ax.set_ylabel("Réel", fontsize=12)
    ax.set_title("Matrice de Confusion", fontsize=14)
    
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()
    print("  ✓ confusion_matrix.png")


def run_evaluation():
    """Lance l'évaluation complète."""
    print("=" * 60)
    print("ÉVALUATION COMPLÈTE")
    print("=" * 60)
    
    print("\n1. Courbes de pretraining SSL :")
    plot_ssl_training_curves()
    
    print("\n2. Courbes de fine-tuning :")
    plot_finetune_curves()
    
    print("\n3. Métriques et graphiques :")
    compute_metrics()
    
    print(f"\n" + "=" * 60)
    print(f"Graphiques sauvés dans : {FIGURES_DIR}/")
    print(f"Métriques sauvées dans : {RESULTS_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    run_evaluation()
