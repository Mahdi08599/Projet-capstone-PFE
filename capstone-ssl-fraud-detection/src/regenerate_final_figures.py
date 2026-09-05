"""
=================================================================
Regeneration des figures du modele final (XGBoost tune)
=================================================================
Contexte : final_model.py entraine un XGBoost de base (AUC 0.9494)
et genere les figures final_*.png. Le modele reellement retenu vient
de hyperparameter_tuning.py (AUC 0.9718, seuil 0.56) mais ce script
ne produit aucune figure. Les PNG etaient donc restes sur l'ancien
modele.

Ce script regenere les 4 figures a partir des predictions reelles du
modele tune (reports/results/tuned_predictions.npz), avec exactement
la meme logique metier que final_model.py.

Usage :
    python src/regenerate_final_figures.py
=================================================================
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIGURES_DIR = os.path.join(ROOT, "reports", "figures")
RESULTS_DIR = os.path.join(ROOT, "reports", "results")
PREDICTIONS = os.path.join(RESULTS_DIR, "tuned_predictions.npz")

# Hypotheses metier (identiques a final_model.py)
AVG_FRAUD_AMOUNT = 149.0
COST_INVESTIGATION = 15.0

SEUIL_RETENU = 0.56  # seuil du modele final (meilleur F1)


def load_predictions():
    data = np.load(PREDICTIONS)
    return data["y_true"].astype(int), data["y_proba"].astype(float)


def roc_curve(y_true, y_proba):
    """Courbe ROC + AUC (equivalent sklearn, sans dependance)."""
    order = np.argsort(-y_proba)
    y = y_true[order]
    tps = np.cumsum(y)
    fps = np.cumsum(1 - y)
    n_pos, n_neg = y.sum(), len(y) - y.sum()
    tpr = np.concatenate([[0.0], tps / n_pos])
    fpr = np.concatenate([[0.0], fps / n_neg])
    auc = np.trapezoid(tpr, fpr)
    return fpr, tpr, auc


def analyze_thresholds(y_true, y_proba):
    """Precision / recall / F1 / impact financier pour chaque seuil."""
    rows = []
    for t in np.arange(0.05, 0.96, 0.01):
        y_pred = y_proba >= t
        tp = int((y_pred & (y_true == 1)).sum())
        fp = int((y_pred & (y_true == 0)).sum())
        fn = int((~y_pred & (y_true == 1)).sum())
        tn = int((~y_pred & (y_true == 0)).sum())

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        money_saved = tp * AVG_FRAUD_AMOUNT
        investigation_cost = (tp + fp) * COST_INVESTIGATION

        rows.append({
            "threshold": round(float(t), 2),
            "precision": prec, "recall": rec, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "money_saved": money_saved,
            "investigation_cost": investigation_cost,
            "net_benefit": money_saved - investigation_cost,
        })
    return rows


def col(rows, key):
    return np.array([r[key] for r in rows])


def row_at(rows, threshold):
    return min(rows, key=lambda r: abs(r["threshold"] - threshold))


def plot_roc(y_true, y_proba):
    fpr, tpr, auc = roc_curve(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(fpr, tpr, linewidth=2.5,
            label=f"XGBoost optimise (AUC = {auc:.4f})", color="#2196F3")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve - Modele Final", fontsize=14)
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "final_roc.png"), dpi=150)
    plt.close(fig)
    print(f"  final_roc.png (AUC = {auc:.4f})")
    return auc


def plot_threshold_analysis(rows):
    t = col(rows, "threshold")
    retenu = row_at(rows, SEUIL_RETENU)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t, col(rows, "precision"), "-", label="Precision", linewidth=2, color="#2196F3")
    ax.plot(t, col(rows, "recall"), "-", label="Recall", linewidth=2, color="#E53935")
    ax.plot(t, col(rows, "f1"), "-", label="F1-score", linewidth=2.5, color="#4CAF50")

    ax.axvline(x=retenu["threshold"], linestyle="--", color="red", alpha=0.6)
    ax.annotate(
        f"Seuil retenu: {retenu['threshold']:.2f}\n"
        f"F1={retenu['f1']:.4f}  P={retenu['precision']:.4f}  R={retenu['recall']:.4f}",
        xy=(retenu["threshold"], retenu["f1"]),
        xytext=(retenu["threshold"] + 0.12, retenu["f1"] - 0.22),
        fontsize=9, ha="left",
        bbox=dict(boxstyle="round", facecolor="lightyellow", edgecolor="black"),
        arrowprops=dict(arrowstyle="->", color="red", alpha=0.7),
    )

    ax.set_xlabel("Seuil de decision", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Precision / Recall / F1 selon le seuil", fontsize=14)
    ax.legend(fontsize=11, loc="lower left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "final_threshold_analysis.png"), dpi=150)
    plt.close(fig)
    print(f"  final_threshold_analysis.png (F1 max au seuil {retenu['threshold']:.2f})")


def plot_business_impact(rows):
    t = col(rows, "threshold")
    retenu = row_at(rows, SEUIL_RETENU)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t, col(rows, "money_saved") / 1000, "-",
            label="Argent sauve ($K)", linewidth=2, color="#4CAF50")
    ax.plot(t, col(rows, "investigation_cost") / 1000, "-",
            label="Cout total investigations ($K)", linewidth=2, color="#FF9800")
    ax.plot(t, col(rows, "net_benefit") / 1000, "-",
            label="Benefice net ($K)", linewidth=2.5, color="#2196F3")

    ax.axvline(x=retenu["threshold"], linestyle="--", color="red", alpha=0.6)
    ax.annotate(
        f"Seuil retenu: {retenu['threshold']:.2f}\nBenefice: ${retenu['net_benefit']:,.0f}",
        xy=(retenu["threshold"], retenu["net_benefit"] / 1000),
        xytext=(retenu["threshold"] + 0.12, retenu["net_benefit"] / 1000 - 60),
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="lightyellow", edgecolor="black"),
        arrowprops=dict(arrowstyle="->", color="red", alpha=0.7),
    )

    ax.set_xlabel("Seuil de decision", fontsize=12)
    ax.set_ylabel("Montant ($K)", fontsize=12)
    ax.set_title("Impact financier selon le seuil de decision", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "final_business_impact.png"), dpi=150)
    plt.close(fig)
    print(f"  final_business_impact.png (benefice ${retenu['net_benefit']:,.0f} au seuil {retenu['threshold']:.2f})")


def plot_confusion_scenarios(y_true, y_proba, rows):
    best_profit = max(rows, key=lambda r: r["net_benefit"])
    recall80 = max([r for r in rows if r["recall"] >= 0.80], key=lambda r: r["f1"])

    scenarios = [
        ("Seuil retenu (F1 max)", row_at(rows, SEUIL_RETENU)),
        ("Recall >= 80%", recall80),
        ("Max profit", best_profit),
    ]

    fig, axes = plt.subplots(1, len(scenarios), figsize=(6 * len(scenarios), 5))
    for ax, (name, r) in zip(axes, scenarios):
        cm = np.array([[r["tn"], r["fp"]], [r["fn"], r["tp"]]])
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,d}", ha="center", va="center",
                        fontsize=13,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks([0, 1], ["Legitime", "Fraude"])
        ax.set_yticks([0, 1], ["Legitime", "Fraude"])
        ax.set_title(f"{name}\n(seuil={r['threshold']:.2f}, F1={r['f1']:.3f})", fontsize=11)
        ax.set_xlabel("Predit")
        ax.set_ylabel("Reel")

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "final_confusion_scenarios.png"), dpi=150)
    plt.close(fig)
    print("  final_confusion_scenarios.png (seuils %s)" %
          ", ".join(f"{r['threshold']:.2f}" for _, r in scenarios))


def run():
    print("=" * 65)
    print("  REGENERATION DES FIGURES - MODELE FINAL TUNE")
    print("=" * 65)

    y_true, y_proba = load_predictions()
    print(f"\nPredictions : {len(y_true):,} lignes de validation, "
          f"{int(y_true.sum()):,} fraudes")

    rows = analyze_thresholds(y_true, y_proba)
    retenu = row_at(rows, SEUIL_RETENU)
    print(f"Seuil {retenu['threshold']:.2f} : "
          f"P={retenu['precision']:.4f} R={retenu['recall']:.4f} "
          f"F1={retenu['f1']:.4f} TP={retenu['tp']:,} FP={retenu['fp']:,} "
          f"Benefice=${retenu['net_benefit']:,.0f}")

    print("\nFigures generees :")
    plot_roc(y_true, y_proba)
    plot_threshold_analysis(rows)
    plot_business_impact(rows)
    plot_confusion_scenarios(y_true, y_proba, rows)

    print("\nTermine.")


if __name__ == "__main__":
    run()
