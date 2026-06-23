"""
=================================================================
Optimisation des hyperparamètres XGBoost
=================================================================
Recherche systématique des meilleurs hyperparamètres avec
RandomizedSearchCV optimisé pour l'average precision (PR-AUC),
la bonne métrique pour les données déséquilibrées.

Usage :
    python src/hyperparameter_tuning.py
=================================================================
"""

import os
import time
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, make_scorer,
)
from scipy.stats import uniform, randint

DATA_DIR = os.path.join("..", "data", "processed")
RESULTS_DIR = os.path.join("..", "reports", "results")
MODEL_DIR = os.path.join("..", "models")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
SEED = 42


def run():
    print("=" * 65)
    print("  OPTIMISATION DES HYPERPARAMÈTRES")
    print("=" * 65)

    # 1. Charger
    print("\n1. Chargement des données V2...")
    train_df = pd.read_parquet(f"{DATA_DIR}/train_clean_v2.parquet")
    val_df = pd.read_parquet(f"{DATA_DIR}/val_clean_v2.parquet")

    y_train = train_df["isFraud"].values
    X_train = train_df.drop(columns=["isFraud"]).values
    y_val = val_df["isFraud"].values
    X_val = val_df.drop(columns=["isFraud"]).values

    ratio = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"   Train: {len(X_train):,} | Val: {len(X_val):,}")
    print(f"   scale_pos_weight = {ratio:.1f}")

    # 2. Espace de recherche
    print("\n2. Définition de l'espace de recherche...")
    param_dist = {
        "n_estimators": randint(300, 800),
        "max_depth": randint(4, 10),
        "learning_rate": uniform(0.01, 0.15),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.6, 0.4),
        "min_child_weight": randint(1, 10),
        "gamma": uniform(0, 0.5),
        "reg_alpha": uniform(0, 1.0),
        "reg_lambda": uniform(0.5, 2.0),
    }

    base_model = xgb.XGBClassifier(
        scale_pos_weight=ratio,
        random_state=SEED,
        eval_metric="aucpr",
        tree_method="hist",  # plus rapide
        verbosity=0,
    )

    # 3. Recherche (optimise average precision = PR-AUC)
    print("\n3. RandomizedSearchCV (30 combinaisons, 3-fold CV)...")
    print("   Métrique optimisée : Average Precision (PR-AUC)")
    print("   Ça peut prendre 30-60 min, patience...\n")

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    scorer = make_scorer(average_precision_score, response_method="predict_proba")

    search = RandomizedSearchCV(
        base_model,
        param_distributions=param_dist,
        n_iter=30,
        scoring=scorer,
        cv=cv,
        random_state=SEED,
        n_jobs=-1,
        verbose=2,
    )

    t0 = time.time()
    search.fit(X_train, y_train)
    elapsed = time.time() - t0

    print(f"\n   ✓ Recherche terminée en {elapsed/60:.1f} min")
    print(f"\n   Meilleurs paramètres :")
    for k, v in search.best_params_.items():
        print(f"     {k:20s} : {v}")
    print(f"   Best CV PR-AUC : {search.best_score_:.4f}")

    # 4. Évaluer le meilleur modèle
    print("\n4. Évaluation sur validation...")
    best_model = search.best_estimator_
    y_proba = best_model.predict_proba(X_val)[:, 1]

    auc = roc_auc_score(y_val, y_proba)
    ap = average_precision_score(y_val, y_proba)

    # Meilleur F1
    best_f1, best_t = 0, 0.5
    for t in np.arange(0.05, 0.95, 0.01):
        f1 = f1_score(y_val, (y_proba >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    y_pred = (y_proba >= best_t).astype(int)
    prec = precision_score(y_val, y_pred, zero_division=0)
    rec = recall_score(y_val, y_pred, zero_division=0)

    print(f"\n{'='*65}")
    print(f"  RÉSULTATS APRÈS OPTIMISATION")
    print(f"{'='*65}")
    print(f"  {'':20s} {'Avant':>12} {'Après':>12} {'Gain':>10}")
    print(f"  {'-'*55}")
    print(f"  {'AUC-ROC':20s} {0.9494:>12.4f} {auc:>12.4f} {auc-0.9494:>+10.4f}")
    print(f"  {'Avg Precision':20s} {0.7020:>12.4f} {ap:>12.4f} {ap-0.7020:>+10.4f}")
    print(f"  {'F1-score':20s} {0.6660:>12.4f} {best_f1:>12.4f} {best_f1-0.6660:>+10.4f}")
    print(f"  {'Precision':20s} {0.7340:>12.4f} {prec:>12.4f} {prec-0.7340:>+10.4f}")
    print(f"  {'Recall':20s} {0.6100:>12.4f} {rec:>12.4f} {rec-0.6100:>+10.4f}")
    print(f"  Seuil optimal : {best_t:.2f}")

    # 5. Sauvegarder
    best_model.save_model(os.path.join(MODEL_DIR, "xgboost_tuned.json"))
    np.savez(
        os.path.join(RESULTS_DIR, "tuned_predictions.npz"),
        y_true=y_val, y_proba=y_proba,
    )

    with open(os.path.join(RESULTS_DIR, "hyperparameter_results.txt"), "w", encoding="utf-8") as f:
        f.write("HYPERPARAMETER TUNING RESULTS\n")
        f.write("=" * 50 + "\n\n")
        f.write("Best params:\n")
        for k, v in search.best_params_.items():
            f.write(f"  {k}: {v}\n")
        f.write(f"\nAUC-ROC      : {auc:.4f}\n")
        f.write(f"Avg Precision: {ap:.4f}\n")
        f.write(f"F1           : {best_f1:.4f}\n")
        f.write(f"Precision    : {prec:.4f}\n")
        f.write(f"Recall       : {rec:.4f}\n")
        f.write(f"Seuil        : {best_t:.2f}\n")

    print(f"\n   ✓ Modèle sauvé : xgboost_tuned.json")
    print(f"   ✓ Rapport : hyperparameter_results.txt")

    print(f"\n{'='*65}")
    print("OPTIMISATION TERMINÉE")
    print(f"{'='*65}")


if __name__ == "__main__":
    run()
