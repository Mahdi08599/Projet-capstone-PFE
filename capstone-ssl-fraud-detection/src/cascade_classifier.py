"""
=================================================================
Classifieur en cascade — Système à 3 niveaux + modèle zone grise
=================================================================
Architecture :

  Transaction
      |
  [Modèle 1 : XGBoost global]
      |
      ├── Score > 0.83 → FRAUDE (blocage automatique)
      ├── Score < 0.30 → LÉGITIME (autorisation automatique)
      └── 0.30 - 0.83 → [Modèle 2 : spécialisé zone grise]
                              |
                              ├── FRAUDE
                              └── LÉGITIME

Le modèle 2 est entraîné UNIQUEMENT sur les cas difficiles
(ceux que le modèle 1 n'arrive pas à trancher clairement).
Il donne une réponse binaire : fraude ou pas.

Usage :
    python src/cascade_classifier.py
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
from sklearn.metrics import (
    classification_report, confusion_matrix,
    recall_score, precision_score, f1_score, roc_auc_score,
)

DATA_DIR = os.path.join("..", "data", "processed")
FIGURES_DIR = os.path.join("..", "reports", "figures")
RESULTS_DIR = os.path.join("..", "reports", "results")
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED = 42
SEUIL_HAUT = 0.83   # au-dessus = fraude automatique
SEUIL_BAS = 0.30    # en-dessous = légitime automatique


def run():
    print("=" * 65)
    print("  CLASSIFIEUR EN CASCADE")
    print("=" * 65)

    # ── 1. Charger les données ───────────────────────────────
    print("\n1. Chargement...")
    train_df = pd.read_parquet(f"{DATA_DIR}/train_clean_v2.parquet")
    val_df = pd.read_parquet(f"{DATA_DIR}/val_clean_v2.parquet")

    y_train = train_df["isFraud"].values
    X_train = train_df.drop(columns=["isFraud"]).values
    y_val = val_df["isFraud"].values
    X_val = val_df.drop(columns=["isFraud"]).values

    n_feat = X_train.shape[1]
    print(f"   {len(X_train):,} train | {len(X_val):,} val | {n_feat} features")

    # ── 2. Modèle 1 : XGBoost global ────────────────────────
    print("\n2. Modèle 1 — XGBoost global...")
    ratio = (y_train == 0).sum() / (y_train == 1).sum()

    model1 = xgb.XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        scale_pos_weight=ratio, random_state=SEED,
        eval_metric="aucpr", early_stopping_rounds=20, verbosity=0,
    )
    model1.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    scores_train = model1.predict_proba(X_train)[:, 1]
    scores_val = model1.predict_proba(X_val)[:, 1]
    print(f"   ✓ AUC global : {roc_auc_score(y_val, scores_val):.4f}")

    # ── 3. Identifier la zone grise ──────────────────────────
    print(f"\n3. Zone grise : scores entre {SEUIL_BAS} et {SEUIL_HAUT}")

    # Sur le train
    mask_gray_train = (scores_train >= SEUIL_BAS) & (scores_train <= SEUIL_HAUT)
    X_gray_train = X_train[mask_gray_train]
    y_gray_train = y_train[mask_gray_train]

    # Sur la validation
    mask_gray_val = (scores_val >= SEUIL_BAS) & (scores_val <= SEUIL_HAUT)
    X_gray_val = X_val[mask_gray_val]
    y_gray_val = y_val[mask_gray_val]

    n_gray_train = len(y_gray_train)
    n_gray_fraud_train = y_gray_train.sum()
    n_gray_val = len(y_gray_val)
    n_gray_fraud_val = y_gray_val.sum()

    print(f"   Train zone grise : {n_gray_train:,} transactions ({n_gray_fraud_train:,} fraudes, {n_gray_fraud_train/n_gray_train*100:.1f}%)")
    print(f"   Val zone grise   : {n_gray_val:,} transactions ({n_gray_fraud_val:,} fraudes, {n_gray_fraud_val/n_gray_val*100:.1f}%)")

    # ── 4. Modèle 2 : spécialisé zone grise ─────────────────
    print("\n4. Modèle 2 — Spécialisé zone grise...")

    ratio_gray = (y_gray_train == 0).sum() / max((y_gray_train == 1).sum(), 1)

    model2 = xgb.XGBClassifier(
        n_estimators=400, max_depth=7, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.7, min_child_weight=3,
        scale_pos_weight=ratio_gray * 1.5,  # encore plus agressif sur la fraude
        random_state=SEED, eval_metric="aucpr",
        early_stopping_rounds=15, verbosity=0,
    )
    model2.fit(X_gray_train, y_gray_train,
               eval_set=[(X_gray_val, y_gray_val)], verbose=False)

    gray_preds = model2.predict(X_gray_val)
    gray_proba = model2.predict_proba(X_gray_val)[:, 1]

    # Optimiser le seuil du modèle 2 pour maximiser le recall
    best_t2, best_recall2 = 0.5, 0
    for t in np.arange(0.10, 0.90, 0.01):
        rec = recall_score(y_gray_val, (gray_proba >= t).astype(int), zero_division=0)
        prec = precision_score(y_gray_val, (gray_proba >= t).astype(int), zero_division=0)
        # On veut au moins 15% de precision pour ne pas noyer les analystes
        if rec > best_recall2 and prec >= 0.15:
            best_recall2 = rec
            best_t2 = t

    print(f"   Seuil optimal modèle 2 : {best_t2:.2f}")
    gray_final = (gray_proba >= best_t2).astype(int)

    # ── 5. Assembler le système complet ──────────────────────
    print("\n5. Assemblage du système complet sur la validation...")

    # Décision finale pour chaque transaction
    final_pred = np.zeros(len(y_val), dtype=int)

    # Zone haute : fraude automatique
    mask_high = scores_val > SEUIL_HAUT
    final_pred[mask_high] = 1

    # Zone basse : légitime automatique
    mask_low = scores_val < SEUIL_BAS
    final_pred[mask_low] = 0

    # Zone grise : modèle 2 décide
    gray_indices = np.where(mask_gray_val)[0]
    final_pred[gray_indices] = gray_final

    # ── 6. Résultats ─────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  RÉSULTATS DU SYSTÈME EN CASCADE")
    print(f"{'='*65}")

    total = len(y_val)
    n_high = mask_high.sum()
    n_low = mask_low.sum()
    n_gray = mask_gray_val.sum()

    print(f"\n  Distribution des décisions :")
    print(f"    Blocage auto (>{SEUIL_HAUT})  : {n_high:>7,} transactions ({n_high/total*100:.1f}%)")
    print(f"    Zone grise ({SEUIL_BAS}-{SEUIL_HAUT})  : {n_gray:>7,} transactions ({n_gray/total*100:.1f}%)")
    print(f"    Autorisé auto (<{SEUIL_BAS})  : {n_low:>7,} transactions ({n_low/total*100:.1f}%)")

    # Métriques globales
    total_frauds = y_val.sum()
    tp = ((final_pred == 1) & (y_val == 1)).sum()
    fp = ((final_pred == 1) & (y_val == 0)).sum()
    fn = ((final_pred == 0) & (y_val == 1)).sum()
    tn = ((final_pred == 0) & (y_val == 0)).sum()

    global_recall = tp / (tp + fn)
    global_precision = tp / (tp + fp)
    global_f1 = 2 * global_precision * global_recall / (global_precision + global_recall)

    print(f"\n  Métriques globales du système :")
    print(f"    Fraudes détectées : {tp:,} / {int(total_frauds):,} ({global_recall*100:.1f}%)")
    print(f"    Fraudes ratées    : {fn:,} ({fn/total_frauds*100:.1f}%)")
    print(f"    Fausses alertes   : {fp:,}")
    print(f"    Precision         : {global_precision:.4f}")
    print(f"    Recall            : {global_recall:.4f}")
    print(f"    F1-score          : {global_f1:.4f}")

    # Détail par zone
    print(f"\n  Détail par zone :")

    # Zone haute
    h_tp = ((scores_val > SEUIL_HAUT) & (y_val == 1)).sum()
    h_fp = ((scores_val > SEUIL_HAUT) & (y_val == 0)).sum()
    print(f"    Zone haute  : {h_tp} fraudes bloquées, {h_fp} faux positifs")

    # Zone grise
    g_tp = ((final_pred == 1) & (y_val == 1) & mask_gray_val).sum()
    g_fn = ((final_pred == 0) & (y_val == 1) & mask_gray_val).sum()
    g_fp = ((final_pred == 1) & (y_val == 0) & mask_gray_val).sum()
    print(f"    Zone grise  : {g_tp} fraudes attrapées par modèle 2, {g_fn} manquées, {g_fp} faux positifs")

    # Zone basse
    l_fn = ((scores_val < SEUIL_BAS) & (y_val == 1)).sum()
    print(f"    Zone basse  : {l_fn} fraudes échappent au système")

    # Impact financier
    avg_fraud = 149.0
    cost_inv = 15.0
    saved = tp * avg_fraud
    lost = fn * avg_fraud
    inv_cost = (tp + fp) * cost_inv
    net = saved - inv_cost

    print(f"\n  Impact financier :")
    print(f"    Argent sauvé        : ${saved:,.0f}")
    print(f"    Pertes résiduelles  : ${lost:,.0f}")
    print(f"    Coût investigations : ${inv_cost:,.0f}")
    print(f"    Bénéfice net        : ${net:,.0f}")

    # ── 7. Comparaison avec modèle simple ────────────────────
    print(f"\n  Comparaison avec seuil unique (F1 optimal = 0.83) :")
    simple_pred = (scores_val > 0.83).astype(int)
    simple_tp = ((simple_pred == 1) & (y_val == 1)).sum()
    simple_fn = ((simple_pred == 0) & (y_val == 1)).sum()
    simple_fp = ((simple_pred == 1) & (y_val == 0)).sum()
    simple_rec = simple_tp / total_frauds
    simple_prec = simple_tp / (simple_tp + simple_fp) if (simple_tp + simple_fp) > 0 else 0
    simple_f1 = 2*simple_prec*simple_rec/(simple_prec+simple_rec) if (simple_prec+simple_rec) > 0 else 0

    print(f"    {'Métrique':<20} {'Seuil unique':>15} {'Cascade':>15} {'Gain':>10}")
    print(f"    {'-'*60}")
    print(f"    {'Fraudes détectées':<20} {simple_tp:>15,} {tp:>15,} {'+' if tp>simple_tp else ''}{tp-simple_tp:>9,}")
    print(f"    {'Fraudes ratées':<20} {simple_fn:>15,} {fn:>15,} {fn-simple_fn:>+9,}")
    print(f"    {'Recall':<20} {simple_rec:>15.4f} {global_recall:>15.4f} {global_recall-simple_rec:>+10.4f}")
    print(f"    {'Precision':<20} {simple_prec:>15.4f} {global_precision:>15.4f} {global_precision-simple_prec:>+10.4f}")
    print(f"    {'F1':<20} {simple_f1:>15.4f} {global_f1:>15.4f} {global_f1-simple_f1:>+10.4f}")

    # ── 8. Graphiques ────────────────────────────────────────
    print(f"\n8. Graphiques...")

    # Confusion matrix du système complet
    cm = confusion_matrix(y_val, final_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues",
                xticklabels=["Legitime", "Fraude"],
                yticklabels=["Legitime", "Fraude"], ax=ax)
    ax.set_xlabel("Predit", fontsize=12)
    ax.set_ylabel("Reel", fontsize=12)
    ax.set_title(f"Matrice de Confusion — Systeme en Cascade", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "cascade_confusion.png"), dpi=150)
    plt.close()
    print("   ✓ cascade_confusion.png")

    # Distribution des scores avec zones
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(scores_val[y_val == 0], bins=100, alpha=0.6, label="Legitime", color="#2196F3", density=True)
    ax.hist(scores_val[y_val == 1], bins=100, alpha=0.6, label="Fraude", color="#E53935", density=True)
    ax.axvline(x=SEUIL_BAS, color="green", linestyle="--", linewidth=2, label=f"Seuil bas ({SEUIL_BAS})")
    ax.axvline(x=SEUIL_HAUT, color="red", linestyle="--", linewidth=2, label=f"Seuil haut ({SEUIL_HAUT})")
    ax.axvspan(0, SEUIL_BAS, alpha=0.05, color="green")
    ax.axvspan(SEUIL_BAS, SEUIL_HAUT, alpha=0.05, color="orange")
    ax.axvspan(SEUIL_HAUT, 1, alpha=0.05, color="red")
    ax.text(SEUIL_BAS/2, ax.get_ylim()[1]*0.9, "AUTORISE", ha="center", fontsize=10, fontweight="bold", color="green")
    ax.text((SEUIL_BAS+SEUIL_HAUT)/2, ax.get_ylim()[1]*0.9, "ZONE GRISE\n(Modele 2)", ha="center", fontsize=10, fontweight="bold", color="orange")
    ax.text((SEUIL_HAUT+1)/2, ax.get_ylim()[1]*0.9, "BLOQUE", ha="center", fontsize=10, fontweight="bold", color="red")
    ax.set_xlabel("Score du modele 1", fontsize=12)
    ax.set_ylabel("Densite", fontsize=12)
    ax.set_title("Distribution des scores — Systeme en Cascade", fontsize=14)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "cascade_zones.png"), dpi=150)
    plt.close()
    print("   ✓ cascade_zones.png")

    # Comparaison barplot
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    labels = ["Seuil unique", "Cascade"]

    axes[0].bar(labels, [simple_rec, global_recall], color=["#888888", "#4CAF50"])
    axes[0].set_title("Recall", fontsize=12, fontweight="bold")
    axes[0].set_ylim(0, 1)
    for i, v in enumerate([simple_rec, global_recall]):
        axes[0].text(i, v+0.02, f"{v:.3f}", ha="center", fontsize=11)

    axes[1].bar(labels, [simple_prec, global_precision], color=["#888888", "#2196F3"])
    axes[1].set_title("Precision", fontsize=12, fontweight="bold")
    axes[1].set_ylim(0, 1)
    for i, v in enumerate([simple_prec, global_precision]):
        axes[1].text(i, v+0.02, f"{v:.3f}", ha="center", fontsize=11)

    axes[2].bar(labels, [simple_f1, global_f1], color=["#888888", "#FF9800"])
    axes[2].set_title("F1-score", fontsize=12, fontweight="bold")
    axes[2].set_ylim(0, 1)
    for i, v in enumerate([simple_f1, global_f1]):
        axes[2].text(i, v+0.02, f"{v:.3f}", ha="center", fontsize=11)

    fig.suptitle("Seuil unique vs Systeme en Cascade", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "cascade_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("   ✓ cascade_comparison.png")

    # Sauvegarder
    with open(os.path.join(RESULTS_DIR, "cascade_report.txt"), "w", encoding="utf-8") as f:
        f.write("SYSTEME EN CASCADE — RAPPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Seuil haut : {SEUIL_HAUT}\n")
        f.write(f"Seuil bas  : {SEUIL_BAS}\n\n")
        f.write(f"Recall global    : {global_recall:.4f}\n")
        f.write(f"Precision globale: {global_precision:.4f}\n")
        f.write(f"F1 global        : {global_f1:.4f}\n\n")
        f.write(f"Fraudes detectees : {tp} / {int(total_frauds)}\n")
        f.write(f"Fraudes ratees    : {fn}\n")
        f.write(f"Benefice net      : ${net:,.0f}\n")
    print("   ✓ cascade_report.txt")

    print(f"\n{'='*65}")
    print("SYSTEME EN CASCADE TERMINE")
    print(f"{'='*65}")


if __name__ == "__main__":
    run()
