"""
=================================================================
Modèle Final — XGBoost optimisé pour la détection de fraude
=================================================================
Objectif : un modèle défendable devant le jury avec des résultats
alignés sur la problématique métier.

Améliorations :
  1. XGBoost avec scale_pos_weight (déséquilibre)
  2. Optimisation du seuil par analyse coût métier
  3. Analyse precision-recall à différents seuils
  4. Résultats interprétables pour le jury

Usage :
    pip install xgboost
    python src/final_model.py
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
from sklearn.metrics import (
    roc_auc_score, roc_curve, f1_score, recall_score,
    precision_score, confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score,
)

try:
    import xgboost as xgb
    USE_XGB = True
    print("XGBoost natif disponible")
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    USE_XGB = False
    print("XGBoost non installe, utilisation de sklearn GradientBoosting")

DATA_DIR = os.path.join("..", "data", "processed")
FIGURES_DIR = os.path.join("..", "reports", "figures")
RESULTS_DIR = os.path.join("..", "reports", "results")
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED = 42


def train_xgboost(X_train, y_train, X_val, y_val):
    """Entraîne XGBoost avec gestion du déséquilibre."""
    n_legit = (y_train == 0).sum()
    n_fraud = (y_train == 1).sum()
    ratio = n_legit / n_fraud

    print(f"  Classes : {n_legit:,} légitimes / {n_fraud:,} fraudes")
    print(f"  scale_pos_weight = {ratio:.1f}")

    if USE_XGB:
        model = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            scale_pos_weight=ratio,
            random_state=SEED,
            eval_metric="aucpr",
            early_stopping_rounds=20,
            verbosity=1,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=50,
        )
    else:
        model = GradientBoostingClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_leaf=30,
            random_state=SEED,
        )
        model.fit(X_train, y_train)

    return model


def analyze_thresholds(y_true, y_proba):
    """Analyse complète des seuils : precision, recall, F1, impact métier."""
    results = []

    # Hypothèses métier
    avg_fraud_amount = 149.0  # montant moyen d'une fraude
    cost_investigation = 15.0  # coût pour investiguer un cas

    for t in np.arange(0.05, 0.96, 0.01):
        y_pred = (y_proba >= t).astype(int)
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        tn = ((y_pred == 0) & (y_true == 0)).sum()

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

        # Impact métier
        money_saved = tp * avg_fraud_amount
        money_lost = fn * avg_fraud_amount
        investigation_cost = (tp + fp) * cost_investigation
        net_benefit = money_saved - investigation_cost

        results.append({
            "threshold": t,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "money_saved": money_saved,
            "money_lost": money_lost,
            "investigation_cost": investigation_cost,
            "net_benefit": net_benefit,
        })

    return pd.DataFrame(results)


def find_optimal_thresholds(analysis_df):
    """Trouve les seuils optimaux pour différents objectifs."""
    thresholds = {}

    # 1. Meilleur F1
    best_f1_idx = analysis_df["f1"].idxmax()
    thresholds["best_f1"] = analysis_df.loc[best_f1_idx]

    # 2. Meilleur bénéfice net
    best_net_idx = analysis_df["net_benefit"].idxmax()
    thresholds["best_profit"] = analysis_df.loc[best_net_idx]

    # 3. Recall >= 70% (objectif métier : détecter au moins 70% des fraudes)
    high_recall = analysis_df[analysis_df["recall"] >= 0.70]
    if len(high_recall) > 0:
        best_hr_idx = high_recall["f1"].idxmax()
        thresholds["high_recall"] = analysis_df.loc[best_hr_idx]

    # 4. Recall >= 80%
    very_high_recall = analysis_df[analysis_df["recall"] >= 0.80]
    if len(very_high_recall) > 0:
        best_vhr_idx = very_high_recall["f1"].idxmax()
        thresholds["very_high_recall"] = analysis_df.loc[best_vhr_idx]

    # 5. Precision >= 50% (au moins 1 alerte sur 2 est vraie)
    good_prec = analysis_df[analysis_df["precision"] >= 0.50]
    if len(good_prec) > 0:
        best_gp_idx = good_prec["recall"].idxmax()
        thresholds["good_precision"] = analysis_df.loc[best_gp_idx]

    return thresholds


def plot_results(analysis_df, optimal_thresholds, y_true, y_proba):
    """Génère tous les graphiques pour le jury."""

    # ── 1. Precision-Recall-F1 vs Seuil ─────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(analysis_df["threshold"], analysis_df["precision"], "-",
            label="Precision", linewidth=2, color="#2196F3")
    ax.plot(analysis_df["threshold"], analysis_df["recall"], "-",
            label="Recall", linewidth=2, color="#E53935")
    ax.plot(analysis_df["threshold"], analysis_df["f1"], "-",
            label="F1-score", linewidth=2.5, color="#4CAF50")

    # Marquer les seuils optimaux
    for name, row in optimal_thresholds.items():
        label_map = {
            "best_f1": "Meilleur F1",
            "best_profit": "Max profit",
            "high_recall": "Recall>=70%",
            "very_high_recall": "Recall>=80%",
            "good_precision": "Prec>=50%",
        }
        ax.axvline(x=row["threshold"], linestyle="--", alpha=0.5)
        ax.annotate(label_map.get(name, name),
                    xy=(row["threshold"], row["f1"]),
                    fontsize=8, ha="center",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

    ax.set_xlabel("Seuil de decision", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Precision / Recall / F1 selon le seuil", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "final_threshold_analysis.png"), dpi=150)
    plt.close()
    print("  ✓ final_threshold_analysis.png")

    # ── 2. Impact financier vs Seuil ─────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(analysis_df["threshold"], analysis_df["money_saved"] / 1000, "-",
            label="Argent sauve ($K)", linewidth=2, color="#4CAF50")
    ax.plot(analysis_df["threshold"], analysis_df["investigation_cost"] / 1000, "-",
            label="Cout investigations ($K)", linewidth=2, color="#FF9800")
    ax.plot(analysis_df["threshold"], analysis_df["net_benefit"] / 1000, "-",
            label="Benefice net ($K)", linewidth=2.5, color="#2196F3")

    best_profit = optimal_thresholds.get("best_profit")
    if best_profit is not None:
        ax.axvline(x=best_profit["threshold"], linestyle="--", color="red", alpha=0.5)
        ax.annotate(f"Seuil optimal: {best_profit['threshold']:.2f}\nBenefice: ${best_profit['net_benefit']:,.0f}",
                    xy=(best_profit["threshold"], best_profit["net_benefit"]/1000),
                    fontsize=9, bbox=dict(boxstyle="round", facecolor="lightyellow"))

    ax.set_xlabel("Seuil de decision", fontsize=12)
    ax.set_ylabel("Montant ($K)", fontsize=12)
    ax.set_title("Impact financier selon le seuil de decision", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "final_business_impact.png"), dpi=150)
    plt.close()
    print("  ✓ final_business_impact.png")

    # ── 3. ROC curve ─────────────────────────────────────────
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(fpr, tpr, linewidth=2.5, label=f"XGBoost optimise (AUC = {auc:.4f})", color="#2196F3")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve - Modele Final", fontsize=14)
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "final_roc.png"), dpi=150)
    plt.close()
    print("  ✓ final_roc.png")

    # ── 4. Confusion matrices pour chaque scénario ───────────
    scenarios = {}
    if "best_f1" in optimal_thresholds:
        scenarios["Meilleur F1"] = optimal_thresholds["best_f1"]
    if "high_recall" in optimal_thresholds:
        scenarios["Recall >= 70%"] = optimal_thresholds["high_recall"]
    if "best_profit" in optimal_thresholds:
        scenarios["Max profit"] = optimal_thresholds["best_profit"]

    if scenarios:
        fig, axes = plt.subplots(1, len(scenarios), figsize=(6*len(scenarios), 5))
        if len(scenarios) == 1:
            axes = [axes]

        for ax, (name, row) in zip(axes, scenarios.items()):
            y_pred = (y_proba >= row["threshold"]).astype(int)
            cm = confusion_matrix(y_true, y_pred)
            sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues",
                       xticklabels=["Legitime", "Fraude"],
                       yticklabels=["Legitime", "Fraude"], ax=ax)
            ax.set_title(f"{name}\n(seuil={row['threshold']:.2f}, F1={row['f1']:.3f})", fontsize=11)
            ax.set_xlabel("Predit")
            ax.set_ylabel("Reel")

        fig.tight_layout()
        fig.savefig(os.path.join(FIGURES_DIR, "final_confusion_scenarios.png"), dpi=150)
        plt.close()
        print("  ✓ final_confusion_scenarios.png")


def run():
    print("=" * 65)
    print("  MODÈLE FINAL — XGBoost optimisé fraude bancaire")
    print("=" * 65)

    # 1. Charger données V2
    print("\n1. Chargement des données V2...")
    train_df = pd.read_parquet(f"{DATA_DIR}/train_clean_v2.parquet")
    val_df = pd.read_parquet(f"{DATA_DIR}/val_clean_v2.parquet")

    y_train = train_df["isFraud"].values
    X_train = train_df.drop(columns=["isFraud"]).values
    y_val = val_df["isFraud"].values
    X_val = val_df.drop(columns=["isFraud"]).values

    print(f"  Train: {len(X_train):,} | Val: {len(X_val):,} | Features: {X_train.shape[1]}")

    # 2. Entraîner XGBoost
    print("\n2. Entraînement XGBoost optimisé...")
    t0 = time.time()
    model = train_xgboost(X_train, y_train, X_val, y_val)
    print(f"  ✓ Entraîné en {time.time()-t0:.1f}s")

    # 3. Prédictions
    print("\n3. Prédictions sur validation...")
    if USE_XGB:
        y_proba = model.predict_proba(X_val)[:, 1]
    else:
        y_proba = model.predict_proba(X_val)[:, 1]

    auc = roc_auc_score(y_val, y_proba)
    ap = average_precision_score(y_val, y_proba)
    print(f"  AUC-ROC          : {auc:.4f}")
    print(f"  Average Precision: {ap:.4f}")

    # 4. Analyse des seuils
    print("\n4. Analyse des seuils de décision...")
    analysis = analyze_thresholds(y_val, y_proba)
    optimal = find_optimal_thresholds(analysis)

    print(f"\n{'='*75}")
    print(f"  {'Scénario':<20} {'Seuil':>7} {'Prec':>7} {'Recall':>7} {'F1':>7} {'Fraudes':>10} {'Bénéfice':>12}")
    print(f"{'='*75}")

    n_frauds = y_val.sum()
    for name, row in optimal.items():
        label_map = {
            "best_f1": "Meilleur F1",
            "best_profit": "Max profit",
            "high_recall": "Recall >= 70%",
            "very_high_recall": "Recall >= 80%",
            "good_precision": "Precision >= 50%",
        }
        label = label_map.get(name, name)
        detected = int(row["tp"])
        print(f"  {label:<20} {row['threshold']:>7.2f} {row['precision']:>7.3f} "
              f"{row['recall']:>7.3f} {row['f1']:>7.3f} "
              f"{detected:>5}/{int(n_frauds):>4} "
              f"${row['net_benefit']:>11,.0f}")

    print(f"{'='*75}")

    # 5. Graphiques
    print("\n5. Graphiques...")
    plot_results(analysis, optimal, y_val, y_proba)

    # 6. Sauvegarder le rapport
    report_path = os.path.join(RESULTS_DIR, "final_model_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("MODELE FINAL — RESULTATS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"AUC-ROC: {auc:.4f}\n")
        f.write(f"Average Precision: {ap:.4f}\n\n")
        f.write("SCENARIOS DE SEUIL:\n")
        f.write("-" * 60 + "\n")
        for name, row in optimal.items():
            f.write(f"\n{name}:\n")
            f.write(f"  Seuil     : {row['threshold']:.2f}\n")
            f.write(f"  Precision : {row['precision']:.4f}\n")
            f.write(f"  Recall    : {row['recall']:.4f}\n")
            f.write(f"  F1        : {row['f1']:.4f}\n")
            f.write(f"  Detectees : {int(row['tp'])} / {int(n_frauds)}\n")
            f.write(f"  Faux pos  : {int(row['fp'])}\n")
            f.write(f"  Benefice  : ${row['net_benefit']:,.0f}\n")

        f.write(f"\n\nCLASSIFICATION REPORT (seuil meilleur F1):\n")
        best = optimal["best_f1"]
        y_pred = (y_proba >= best["threshold"]).astype(int)
        f.write(classification_report(y_val, y_pred, target_names=["Legitime", "Fraude"]))

    print(f"  ✓ final_model_report.txt")

    # Sauvegarder prédictions
    np.savez(
        os.path.join(RESULTS_DIR, "final_predictions.npz"),
        y_true=y_val, y_proba=y_proba,
    )

    print(f"\n{'='*65}")
    print("MODELE FINAL TERMINE")
    print(f"{'='*65}")


if __name__ == "__main__":
    run()
