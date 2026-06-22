"""
=================================================================
Évaluation V2 — avec seuil optimisé
=================================================================
Usage :
    python src/evaluate_v2.py
=================================================================
"""

import os
import numpy as np
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

MODEL_DIR = os.path.join("..", "models")
FIGURES_DIR = os.path.join("..", "reports", "figures")
RESULTS_DIR = os.path.join("..", "reports", "results")
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def run():
    # Charger les résultats V2
    path = os.path.join(MODEL_DIR, "finetune_v2_history.npz")
    if not os.path.exists(path):
        print("Lancer d'abord : python finetune_v2.py")
        return

    data = np.load(path)
    y_true = data["val_labels"]
    y_proba = data["val_preds"]
    best_t = float(data["best_threshold"])
    train_losses = data["train_losses"]

    y_pred = (y_proba >= best_t).astype(int)

    # Métriques
    auc = roc_auc_score(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print("=" * 55)
    print("        RÉSULTATS V2 (seuil optimisé)")
    print("=" * 55)
    print(f"  Seuil optimal   : {best_t:.2f}")
    print(f"  AUC-ROC         : {auc:.4f}")
    print(f"  Avg Precision   : {ap:.4f}")
    print(f"  Precision       : {precision:.4f}")
    print(f"  Recall          : {recall:.4f}")
    print(f"  F1-score        : {f1:.4f}")
    print("=" * 55)

    # Sauvegarder
    report = classification_report(y_true, y_pred, target_names=["Légitime", "Fraude"])
    with open(os.path.join(RESULTS_DIR, "metrics_v2.txt"), "w") as f:
        f.write("FRAUD DETECTION RESULTS (V2)\n")
        f.write(f"Threshold : {best_t:.2f}\n")
        f.write(f"AUC-ROC   : {auc:.4f}\n")
        f.write(f"AP        : {ap:.4f}\n")
        f.write(f"Precision : {precision:.4f}\n")
        f.write(f"Recall    : {recall:.4f}\n")
        f.write(f"F1        : {f1:.4f}\n\n")
        f.write(report)
    print("  ✓ metrics_v2.txt")

    # ── Courbe ROC ──────────────────────────────────────────
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(fpr, tpr, linewidth=2.5, label=f"SSL model (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve — Fraud Detection (V2)", fontsize=14)
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "roc_curve_v2.png"), dpi=150)
    plt.close()
    print("  ✓ roc_curve_v2.png")

    # ── Precision-Recall ────────────────────────────────────
    prec_c, rec_c, _ = precision_recall_curve(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(rec_c, prec_c, linewidth=2.5, label=f"AP = {ap:.4f}")
    ax.axvline(x=recall, color="red", linestyle="--", alpha=0.5, label=f"Seuil = {best_t:.2f}")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve (V2)", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "precision_recall_v2.png"), dpi=150)
    plt.close()
    print("  ✓ precision_recall_v2.png")

    # ── Matrice de confusion ────────────────────────────────
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues",
                xticklabels=["Légitime", "Fraude"],
                yticklabels=["Légitime", "Fraude"], ax=ax)
    ax.set_xlabel("Prédit", fontsize=12)
    ax.set_ylabel("Réel", fontsize=12)
    ax.set_title(f"Matrice de Confusion (seuil = {best_t:.2f})", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "confusion_matrix_v2.png"), dpi=150)
    plt.close()
    print("  ✓ confusion_matrix_v2.png")

    # ── Training loss curve ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(train_losses)+1), train_losses, "o-", linewidth=2)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Fine-tuning V2 — Training Loss", fontsize=14)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "finetune_v2_loss.png"), dpi=150)
    plt.close()
    print("  ✓ finetune_v2_loss.png")

    # ── SSL pretraining curve ───────────────────────────────
    ssl_path = os.path.join(MODEL_DIR, "ssl_training_history.npz")
    if os.path.exists(ssl_path):
        ssl_data = np.load(ssl_path)
        fig, ax = plt.subplots(figsize=(8, 5))
        epochs = range(1, len(ssl_data["train_losses"])+1)
        ax.plot(epochs, ssl_data["train_losses"], "o-", label="Train", linewidth=2)
        ax.plot(epochs, ssl_data["val_losses"], "s-", label="Validation", linewidth=2)
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Masked MSE Loss", fontsize=12)
        ax.set_title("SSL Pretraining — Reconstruction Loss", fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(FIGURES_DIR, "ssl_training_curve.png"), dpi=150)
        plt.close()
        print("  ✓ ssl_training_curve.png")

    print(f"\nTout sauvé dans {FIGURES_DIR}/ et {RESULTS_DIR}/")


if __name__ == "__main__":
    run()
