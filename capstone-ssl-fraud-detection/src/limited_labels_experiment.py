"""
=================================================================
Expérience clé : SSL avec labels limités
=================================================================
C'est L'EXPÉRIENCE CENTRALE du mémoire.

Question de recherche :
  "Le SSL améliore-t-il la détection de fraude quand les labels
   sont limités ?"

Protocole :
  On compare XGBoost seul vs SSL+XGBoost en faisant varier
  le nombre de labels disponibles : 1%, 2%, 5%, 10%, 50%, 100%

  Le SSL utilise TOUTES les données (sans labels) pour apprendre.
  Le XGBoost utilise seulement X% des labels pour classifier.

  Si le SSL aide, l'écart sera grand avec peu de labels
  et se réduira quand on a beaucoup de labels.

Usage :
    python src/limited_labels_experiment.py
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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score

from ssl_model import SSLFraudModel

MODEL_DIR = os.path.join("..", "models")
DATA_DIR = os.path.join("..", "data", "processed")
FIGURES_DIR = os.path.join("..", "reports", "figures")
RESULTS_DIR = os.path.join("..", "reports", "results")
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

# Proportions de labels à tester
LABEL_FRACTIONS = [0.01, 0.02, 0.05, 0.10, 0.25, 0.50, 1.0]


def extract_embeddings(model, X, batch_size=1024):
    model.eval()
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.FloatTensor(X[i:i+batch_size]).to(DEVICE)
            z = model.encoder(batch)
            embeddings.append(z.cpu().numpy())
    return np.vstack(embeddings)


def subsample_labels(X, y, fraction, seed=SEED):
    """Sélectionne un sous-ensemble stratifié des données labellisées."""
    rng = np.random.RandomState(seed)
    n = len(y)
    n_sample = max(int(n * fraction), 100)

    # Stratifié : garder le ratio de fraude
    idx_fraud = np.where(y == 1)[0]
    idx_legit = np.where(y == 0)[0]

    n_fraud = max(int(len(idx_fraud) * fraction), 10)
    n_legit = min(n_sample - n_fraud, len(idx_legit))

    selected_fraud = rng.choice(idx_fraud, size=n_fraud, replace=False)
    selected_legit = rng.choice(idx_legit, size=n_legit, replace=False)
    selected = np.concatenate([selected_fraud, selected_legit])

    return X[selected], y[selected]


def train_and_evaluate(X_train, y_train, X_val, y_val):
    """Entraîne un GradientBoosting et retourne les métriques."""
    clf = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        min_samples_leaf=30,
        random_state=SEED,
    )
    clf.fit(X_train, y_train)
    y_proba = clf.predict_proba(X_val)[:, 1]

    auc = roc_auc_score(y_val, y_proba)

    # Trouver le meilleur seuil
    best_f1, best_t = 0, 0.5
    for t in np.arange(0.05, 0.95, 0.05):
        f1 = f1_score(y_val, (y_proba >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    y_pred = (y_proba >= best_t).astype(int)
    rec = recall_score(y_val, y_pred, zero_division=0)
    prec = precision_score(y_val, y_pred, zero_division=0)

    return {"auc": auc, "f1": best_f1, "recall": rec, "precision": prec}


def run():
    print("=" * 60)
    print("EXPÉRIENCE : SSL AVEC LABELS LIMITÉS")
    print("=" * 60)

    # 1. Charger les données
    print("\n1. Chargement...")
    train_df = pd.read_parquet(f"{DATA_DIR}/train_clean.parquet")
    val_df = pd.read_parquet(f"{DATA_DIR}/val_clean.parquet")

    y_train = train_df["isFraud"].values
    X_train = train_df.drop(columns=["isFraud"]).values
    y_val = val_df["isFraud"].values
    X_val = val_df.drop(columns=["isFraud"]).values

    # 2. Charger l'encodeur SSL et extraire les embeddings
    print("2. Extraction des embeddings SSL...")
    model = SSLFraudModel(input_dim=X_train.shape[1])
    model.encoder.load_state_dict(
        torch.load(os.path.join(MODEL_DIR, "ssl_encoder.pt"), weights_only=True)
    )
    model = model.to(DEVICE)

    Z_train = extract_embeddings(model, X_train)
    Z_val = extract_embeddings(model, X_val)

    # Combiner features originales + SSL
    X_train_combined = np.hstack([X_train, Z_train])
    X_val_combined = np.hstack([X_val, Z_val])
    print(f"   ✓ Embeddings extraits")

    # 3. Expérience avec différentes proportions de labels
    print(f"\n3. Comparaison avec {len(LABEL_FRACTIONS)} niveaux de labels...\n")

    results_baseline = []
    results_ssl = []

    for frac in LABEL_FRACTIONS:
        n_labels = max(int(len(y_train) * frac), 100)
        n_frauds = int(sum(y_train) * frac)
        print(f"  [{frac*100:5.1f}% labels] {n_labels:,} samples ({n_frauds} fraudes)...", end="", flush=True)

        t0 = time.time()

        # Sous-échantillonner les labels
        X_sub, y_sub = subsample_labels(X_train, y_train, frac)
        X_sub_combined, _ = subsample_labels(X_train_combined, y_train, frac)

        # Baseline : XGBoost sur features originales
        r_base = train_and_evaluate(X_sub, y_sub, X_val, y_val)
        results_baseline.append(r_base)

        # SSL : XGBoost sur features originales + embeddings SSL
        r_ssl = train_and_evaluate(X_sub_combined, y_sub, X_val_combined, y_val)
        results_ssl.append(r_ssl)

        elapsed = time.time() - t0
        gain_auc = r_ssl["auc"] - r_base["auc"]
        gain_f1 = r_ssl["f1"] - r_base["f1"]
        sign = "+" if gain_auc >= 0 else ""

        print(f"  Baseline AUC={r_base['auc']:.4f} F1={r_base['f1']:.4f} | "
              f"SSL AUC={r_ssl['auc']:.4f} F1={r_ssl['f1']:.4f} | "
              f"Gain: {sign}{gain_auc:.4f} AUC, {'+' if gain_f1>=0 else ''}{gain_f1:.4f} F1 | "
              f"{elapsed:.0f}s")

    # 4. Tableau récapitulatif
    print(f"\n{'='*70}")
    print(f"{'Labels':>8} | {'Baseline AUC':>13} {'F1':>7} | {'SSL AUC':>10} {'F1':>7} | {'Gain AUC':>9} {'Gain F1':>8}")
    print(f"{'-'*70}")
    for i, frac in enumerate(LABEL_FRACTIONS):
        b = results_baseline[i]
        s = results_ssl[i]
        gain_a = s["auc"] - b["auc"]
        gain_f = s["f1"] - b["f1"]
        print(f"{frac*100:7.1f}% | {b['auc']:>12.4f} {b['f1']:>7.4f} | {s['auc']:>9.4f} {s['f1']:>7.4f} | "
              f"{'+' if gain_a>=0 else ''}{gain_a:>8.4f} {'+' if gain_f>=0 else ''}{gain_f:>7.4f}")
    print(f"{'='*70}")

    # 5. Graphiques
    print("\n5. Génération des graphiques...")

    fracs_pct = [f * 100 for f in LABEL_FRACTIONS]

    # ── AUC en fonction du % de labels ──────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].plot(fracs_pct, [r["auc"] for r in results_baseline], "o-",
                 label="Baseline (XGBoost)", linewidth=2.5, markersize=8, color="#888888")
    axes[0].plot(fracs_pct, [r["auc"] for r in results_ssl], "s-",
                 label="SSL + XGBoost", linewidth=2.5, markersize=8, color="#E53935")
    axes[0].set_xlabel("% de labels utilisés", fontsize=12)
    axes[0].set_ylabel("AUC-ROC", fontsize=12)
    axes[0].set_title("AUC-ROC vs quantité de labels", fontsize=14)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xscale("log")
    axes[0].set_xticks(fracs_pct)
    axes[0].set_xticklabels([f"{f:.0f}%" if f >= 1 else f"{f}%" for f in fracs_pct])

    # ── F1 en fonction du % de labels ───────────────────────
    axes[1].plot(fracs_pct, [r["f1"] for r in results_baseline], "o-",
                 label="Baseline (XGBoost)", linewidth=2.5, markersize=8, color="#888888")
    axes[1].plot(fracs_pct, [r["f1"] for r in results_ssl], "s-",
                 label="SSL + XGBoost", linewidth=2.5, markersize=8, color="#E53935")
    axes[1].set_xlabel("% de labels utilisés", fontsize=12)
    axes[1].set_ylabel("F1-score", fontsize=12)
    axes[1].set_title("F1-score vs quantité de labels", fontsize=14)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xscale("log")
    axes[1].set_xticks(fracs_pct)
    axes[1].set_xticklabels([f"{f:.0f}%" if f >= 1 else f"{f}%" for f in fracs_pct])

    fig.suptitle("Impact du SSL selon la disponibilité des labels", fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "limited_labels_experiment.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("   ✓ limited_labels_experiment.png")

    # ── Gain SSL ────────────────────────────────────────────
    gains_auc = [results_ssl[i]["auc"] - results_baseline[i]["auc"] for i in range(len(LABEL_FRACTIONS))]
    gains_f1 = [results_ssl[i]["f1"] - results_baseline[i]["f1"] for i in range(len(LABEL_FRACTIONS))]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(LABEL_FRACTIONS))
    w = 0.35
    bars1 = ax.bar(x - w/2, gains_auc, w, label="Gain AUC", color="#2196F3", alpha=0.8)
    bars2 = ax.bar(x + w/2, gains_f1, w, label="Gain F1", color="#E53935", alpha=0.8)
    ax.axhline(y=0, color="black", linewidth=0.8, linestyle="-")
    ax.set_xlabel("% de labels utilisés", fontsize=12)
    ax.set_ylabel("Gain apporté par le SSL", fontsize=12)
    ax.set_title("Apport du SSL selon la quantité de labels", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{f*100:.0f}%" if f >= 0.01 else f"{f*100}%" for f in LABEL_FRACTIONS])
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "ssl_gain_by_labels.png"), dpi=150)
    plt.close()
    print("   ✓ ssl_gain_by_labels.png")

    # 6. Sauvegarder
    with open(os.path.join(RESULTS_DIR, "limited_labels_results.txt"), "w") as f:
        f.write("LIMITED LABELS EXPERIMENT\n")
        f.write("=" * 60 + "\n\n")
        for i, frac in enumerate(LABEL_FRACTIONS):
            b = results_baseline[i]
            s = results_ssl[i]
            f.write(f"{frac*100:.1f}% labels:\n")
            f.write(f"  Baseline: AUC={b['auc']:.4f} F1={b['f1']:.4f} Rec={b['recall']:.4f}\n")
            f.write(f"  SSL:      AUC={s['auc']:.4f} F1={s['f1']:.4f} Rec={s['recall']:.4f}\n")
            f.write(f"  Gain:     AUC={s['auc']-b['auc']:+.4f} F1={s['f1']-b['f1']:+.4f}\n\n")
    print("   ✓ limited_labels_results.txt")

    print(f"\n{'='*60}")
    print("CONCLUSION DE L'EXPÉRIENCE :")
    print("Si le SSL aide surtout avec peu de labels,")
    print("cela confirme la thèse du mémoire.")
    print(f"{'='*60}")


if __name__ == "__main__":
    run()
