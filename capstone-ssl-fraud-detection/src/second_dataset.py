"""
=================================================================
Test sur le 2ème dataset : Credit Card Fraud (European)
=================================================================
284,807 transactions européennes, 492 fraudes (0.17%)
Features V1-V28 (PCA anonymisé) + Amount + Time

Objectif : montrer que la méthodologie est généralisable
sur un dataset plus déséquilibré et avec des features différentes.

Usage :
    python src/second_dataset.py
=================================================================
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, f1_score, recall_score, precision_score,
    confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score, roc_curve,
)

FIGURES_DIR = os.path.join("..", "reports", "figures")
RESULTS_DIR = os.path.join("..", "reports", "results")
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
SEED = 42


def run():
    print("=" * 65)
    print("  2ème DATASET : CREDIT CARD FRAUD (EUROPEAN)")
    print("=" * 65)

    # 1. Charger et préparer
    print("\n1. Chargement...")

    # Chercher le fichier dans plusieurs emplacements possibles
    possible_paths = [
        os.path.join("..", "data", "raw", "creditcard.csv"),
        os.path.join("..", "creditcard_data", "creditcard.csv"),
        "creditcard.csv",
    ]
    df = None
    for p in possible_paths:
        if os.path.exists(p):
            df = pd.read_csv(p)
            print(f"   Chargé depuis {p}")
            break

    if df is None:
        print("   ERREUR: creditcard.csv non trouvé")
        print("   Placez le fichier dans data/raw/")
        return

    print(f"   {len(df):,} transactions | {df['Class'].sum()} fraudes ({df['Class'].mean()*100:.3f}%)")

    # 2. Preprocessing
    print("\n2. Preprocessing...")
    y = df["Class"].values
    X = df.drop(columns=["Class"])

    # Normaliser Amount et Time
    scaler = StandardScaler()
    X["Amount_scaled"] = scaler.fit_transform(X[["Amount"]])
    X["Time_scaled"] = scaler.fit_transform(X[["Time"]])
    X["Amount_log"] = np.log1p(X["Amount"])
    X = X.drop(columns=["Amount", "Time"])

    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X.values, y, test_size=0.20, random_state=SEED, stratify=y
    )
    print(f"   Train: {len(X_train):,} ({y_train.sum()} fraudes)")
    print(f"   Val:   {len(X_val):,} ({y_val.sum()} fraudes)")

    # 3. XGBoost
    print("\n3. Entraînement XGBoost...")
    ratio = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"   scale_pos_weight = {ratio:.1f}")

    model = xgb.XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        scale_pos_weight=ratio, random_state=SEED,
        eval_metric="aucpr", early_stopping_rounds=20, verbosity=0,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    y_proba = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_proba)
    ap = average_precision_score(y_val, y_proba)
    print(f"   AUC-ROC          : {auc:.4f}")
    print(f"   Average Precision: {ap:.4f}")

    # 4. Analyse des seuils
    print("\n4. Analyse des seuils...")
    print(f"\n   {'Seuil':>7} {'Prec':>7} {'Recall':>7} {'F1':>7} {'Fraudes':>10} {'FP':>7}")
    print(f"   {'-'*52}")

    best_f1, best_t = 0, 0.5
    for t in np.arange(0.05, 0.95, 0.05):
        y_pred = (y_proba >= t).astype(int)
        tp = ((y_pred == 1) & (y_val == 1)).sum()
        fp = ((y_pred == 1) & (y_val == 0)).sum()
        fn = ((y_pred == 0) & (y_val == 1)).sum()
        prec = tp / (tp + fp) if (tp + fp) else 0
        rec = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0

        if f1 > best_f1:
            best_f1 = f1
            best_t = t

        marker = " <--" if abs(prec - rec) < 0.05 else ""
        print(f"   {t:>7.2f} {prec:>7.3f} {rec:>7.3f} {f1:>7.3f} {tp:>5}/{int(y_val.sum()):>4}  {fp:>6}{marker}")

    # 5. Résultats au meilleur F1
    y_pred_best = (y_proba >= best_t).astype(int)
    tp = ((y_pred_best == 1) & (y_val == 1)).sum()
    fp = ((y_pred_best == 1) & (y_val == 0)).sum()
    fn = ((y_pred_best == 0) & (y_val == 1)).sum()
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    f1 = 2 * prec * rec / (prec + rec)

    print(f"\n{'='*65}")
    print(f"  RÉSULTATS — 2ème DATASET")
    print(f"{'='*65}")
    print(f"  Seuil optimal   : {best_t:.2f}")
    print(f"  AUC-ROC         : {auc:.4f}")
    print(f"  Avg Precision   : {ap:.4f}")
    print(f"  Precision       : {prec:.4f}")
    print(f"  Recall          : {rec:.4f}")
    print(f"  F1-score        : {f1:.4f}")
    print(f"  Fraudes det.    : {tp} / {int(y_val.sum())}")
    print(f"  Fausses alertes : {fp}")

    # 6. Comparaison avec IEEE-CIS
    print(f"\n  COMPARAISON AVEC IEEE-CIS :")
    print(f"  {'':25s} {'IEEE-CIS':>12} {'European':>12}")
    print(f"  {'-'*50}")
    print(f"  {'Transactions':25s} {'590,540':>12} {'284,807':>12}")
    print(f"  {'Taux de fraude':25s} {'3.50%':>12} {'0.17%':>12}")
    print(f"  {'Features':25s} {'743':>12} {'31':>12}")
    print(f"  {'AUC-ROC':25s} {'0.9494':>12} {auc:>12.4f}")
    print(f"  {'F1 (best)':25s} {'0.666':>12} {f1:>12.4f}")
    print(f"  {'Precision':25s} {'0.734':>12} {prec:>12.4f}")
    print(f"  {'Recall':25s} {'0.610':>12} {rec:>12.4f}")

    # 7. Graphiques
    print(f"\n7. Graphiques...")

    # ROC
    fpr, tpr, _ = roc_curve(y_val, y_proba)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(fpr, tpr, linewidth=2.5, label=f"European dataset (AUC={auc:.4f})", color="#E53935")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve - 2eme Dataset (European)", fontsize=14)
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "second_dataset_roc.png"), dpi=150)
    plt.close()
    print("   ✓ second_dataset_roc.png")

    # Confusion matrix
    cm = confusion_matrix(y_val, y_pred_best)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues",
                xticklabels=["Legitime", "Fraude"],
                yticklabels=["Legitime", "Fraude"], ax=ax)
    ax.set_xlabel("Predit", fontsize=12)
    ax.set_ylabel("Reel", fontsize=12)
    ax.set_title(f"Confusion Matrix - European (seuil={best_t:.2f})", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "second_dataset_confusion.png"), dpi=150)
    plt.close()
    print("   ✓ second_dataset_confusion.png")

    # Comparaison barplot
    fig, axes = plt.subplots(1, 4, figsize=(14, 5))
    metrics = [
        ("AUC-ROC", 0.9494, auc),
        ("Precision", 0.734, prec),
        ("Recall", 0.610, rec),
        ("F1-score", 0.666, f1),
    ]
    for ax, (name, ieee, euro) in zip(axes, metrics):
        bars = ax.bar(["IEEE-CIS", "European"], [ieee, euro],
                      color=["#2196F3", "#E53935"], alpha=0.85)
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.set_ylim(0, 1.1)
        for bar, val in zip(bars, [ieee, euro]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"{val:.3f}", ha="center", fontsize=10)
    fig.suptitle("Comparaison des performances sur les 2 datasets", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "datasets_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("   ✓ datasets_comparison.png")

    # Sauvegarder
    with open(os.path.join(RESULTS_DIR, "second_dataset_results.txt"), "w", encoding="utf-8") as f:
        f.write("2eme DATASET : CREDIT CARD FRAUD (EUROPEAN)\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Transactions : {len(df):,}\n")
        f.write(f"Fraudes      : {df['Class'].sum()} ({df['Class'].mean()*100:.3f}%)\n")
        f.write(f"Features     : {X.shape[1]}\n\n")
        f.write(f"AUC-ROC      : {auc:.4f}\n")
        f.write(f"Avg Precision: {ap:.4f}\n")
        f.write(f"Seuil        : {best_t:.2f}\n")
        f.write(f"Precision    : {prec:.4f}\n")
        f.write(f"Recall       : {rec:.4f}\n")
        f.write(f"F1           : {f1:.4f}\n")
    print("   ✓ second_dataset_results.txt")

    print(f"\n{'='*65}")
    print("ANALYSE 2ème DATASET TERMINÉE")
    print(f"{'='*65}")


if __name__ == "__main__":
    run()
