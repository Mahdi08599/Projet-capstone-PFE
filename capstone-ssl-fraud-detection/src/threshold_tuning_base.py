"""
=================================================================
Adaptation du seuil metier pour le modele XGBoost de base
=================================================================
Analyse les predictions du modele optimise afin de choisir un seuil
coherent pour une banque digitale : detection, faux positifs, charge
d'investigation et benefice net.

Prerequis :
    python src/hyperparameter_tuning.py

Usage :
    python src/threshold_tuning_base.py
=================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

RESULTS_DIR = os.path.join("reports", "results")
FIGURES_DIR = os.path.join("reports", "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

PREDICTIONS_PATH = os.path.join(RESULTS_DIR, "tuned_predictions.npz")

# Hypotheses metier simples et defendables pour une banque digitale.
AVG_FRAUD_AMOUNT = 149.0
COST_INVESTIGATION = 15.0
RECOMMENDED_SCENARIO = "best_f1"


def load_predictions():
    if not os.path.exists(PREDICTIONS_PATH):
        raise FileNotFoundError(
            "Predictions introuvables. Lancez d'abord : python src/hyperparameter_tuning.py"
        )

    data = np.load(PREDICTIONS_PATH)
    return data["y_true"], data["y_proba"]


def analyze_thresholds(y_true, y_proba):
    rows = []

    for threshold in np.arange(0.05, 0.96, 0.01):
        y_pred = (y_proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        money_saved = tp * AVG_FRAUD_AMOUNT
        money_lost = fn * AVG_FRAUD_AMOUNT
        investigation_cost = (tp + fp) * COST_INVESTIGATION
        net_benefit = money_saved - investigation_cost

        rows.append({
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "alerts": tp + fp,
            "money_saved": money_saved,
            "money_lost": money_lost,
            "investigation_cost": investigation_cost,
            "net_benefit": net_benefit,
        })

    return pd.DataFrame(rows)


def select_scenarios(analysis):
    scenarios = {}

    scenarios["best_f1"] = analysis.loc[analysis["f1"].idxmax()]
    scenarios["best_profit"] = analysis.loc[analysis["net_benefit"].idxmax()]

    high_recall = analysis[analysis["recall"] >= 0.80]
    if not high_recall.empty:
        scenarios["high_recall_80"] = high_recall.loc[high_recall["f1"].idxmax()]

    strong_precision = analysis[analysis["precision"] >= 0.85]
    if not strong_precision.empty:
        scenarios["precision_85"] = strong_precision.loc[strong_precision["recall"].idxmax()]

    bank_balanced = analysis[
        (analysis["recall"] >= 0.70)
        & (analysis["precision"] >= 0.80)
    ]
    if not bank_balanced.empty:
        scenarios["bank_balanced"] = bank_balanced.loc[bank_balanced["net_benefit"].idxmax()]

    return scenarios


def plot_thresholds(analysis, scenarios):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(analysis["threshold"], analysis["precision"], label="Precision", linewidth=2)
    axes[0].plot(analysis["threshold"], analysis["recall"], label="Recall", linewidth=2)
    axes[0].plot(analysis["threshold"], analysis["f1"], label="F1-score", linewidth=2.5)
    axes[0].set_title("Performance selon le seuil")
    axes[0].set_xlabel("Seuil de decision")
    axes[0].set_ylabel("Score")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        analysis["threshold"],
        analysis["net_benefit"] / 1000,
        label="Benefice net ($K)",
        linewidth=2.5,
        color="#1B7F5A",
    )
    axes[1].plot(
        analysis["threshold"],
        analysis["investigation_cost"] / 1000,
        label="Cout investigation ($K)",
        linewidth=2,
        color="#D97706",
    )
    axes[1].set_title("Impact metier selon le seuil")
    axes[1].set_xlabel("Seuil de decision")
    axes[1].set_ylabel("Montant ($K)")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    for name, row in scenarios.items():
        for ax in axes:
            if name == RECOMMENDED_SCENARIO:
                ax.axvline(row["threshold"], linestyle="-", color="#B91C1C", linewidth=2.2, alpha=0.85)
            else:
                ax.axvline(row["threshold"], linestyle="--", color="gray", alpha=0.35)

    if RECOMMENDED_SCENARIO in scenarios:
        row = scenarios[RECOMMENDED_SCENARIO]
        axes[0].annotate(
            f"Seuil retenu: {row['threshold']:.2f}",
            xy=(row["threshold"], row["f1"]),
            xytext=(row["threshold"] - 0.19, row["f1"] + 0.12),
            arrowprops={"arrowstyle": "->", "color": "#B91C1C"},
            color="#B91C1C",
            fontsize=10,
        )
        axes[1].annotate(
            f"Strategie equilibree\n${row['net_benefit']/1000:.0f}K",
            xy=(row["threshold"], row["net_benefit"] / 1000),
            xytext=(row["threshold"] - 0.18, row["net_benefit"] / 1000 + 70),
            arrowprops={"arrowstyle": "->", "color": "#B91C1C"},
            color="#B91C1C",
            fontsize=10,
        )

    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "base_threshold_tuning.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def format_scenario(name):
    labels = {
        "best_f1": "Meilleur F1",
        "best_profit": "Benefice maximal",
        "high_recall_80": "Recall >= 80%",
        "precision_85": "Precision >= 85%",
        "bank_balanced": "Equilibre banque digitale",
    }
    return labels.get(name, name)


def write_report(y_true, y_proba, analysis, scenarios, figure_path):
    auc = roc_auc_score(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    n_frauds = int(y_true.sum())

    report_path = os.path.join(RESULTS_DIR, "base_threshold_tuning_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("ADAPTATION DU SEUIL - MODELE XGBOOST BASE OPTIMISE\n")
        f.write("=" * 62 + "\n\n")
        f.write(f"AUC-ROC          : {auc:.4f}\n")
        f.write(f"Average Precision: {ap:.4f}\n")
        f.write(f"Fraudes validation: {n_frauds:,}\n")
        f.write(f"Hypothese montant moyen fraude: ${AVG_FRAUD_AMOUNT:.0f}\n")
        f.write(f"Hypothese cout investigation : ${COST_INVESTIGATION:.0f}\n")
        f.write(f"Figure: {figure_path}\n\n")

        if RECOMMENDED_SCENARIO in scenarios:
            row = scenarios[RECOMMENDED_SCENARIO]
            best_profit = scenarios.get("best_profit")
            f.write("SEUIL RETENU POUR LA STRATEGIE BANQUE DIGITALE\n")
            f.write("-" * 62 + "\n")
            f.write(f"Seuil recommande : {row['threshold']:.2f}\n")
            f.write("Logique          : maximiser le F1-score du modele final optimise\n")
            f.write("Lecture metier   : compromis precision / recall avec forte performance operationnelle\n")
            f.write(f"Precision        : {row['precision']:.4f}\n")
            f.write(f"Recall           : {row['recall']:.4f}\n")
            f.write(f"F1-score         : {row['f1']:.4f}\n")
            f.write(f"Faux positifs    : {int(row['fp']):,}\n")
            f.write(f"Alertes totales  : {int(row['alerts']):,}\n")
            f.write(f"Benefice net     : ${row['net_benefit']:,.0f}\n")
            if best_profit is not None:
                benefit_gap = best_profit["net_benefit"] - row["net_benefit"]
                fp_reduction = best_profit["fp"] - row["fp"]
                f.write(f"Ecart vs profit max : -${benefit_gap:,.0f}\n")
                f.write(f"Reduction FP vs profit max : {int(fp_reduction):,}\n")
            f.write("\nInterpretation : ce seuil detecte une part importante des fraudes tout en\n")
            f.write("gardant une precision elevee. Il limite la charge d'investigation et reduit\n")
            f.write("le risque de bloquer inutilement des clients legitimes.\n\n")

        f.write("SCENARIOS DE SEUIL\n")
        f.write("-" * 62 + "\n")
        for name, row in scenarios.items():
            f.write(f"\n{format_scenario(name)}\n")
            f.write(f"  Seuil       : {row['threshold']:.2f}\n")
            f.write(f"  Precision   : {row['precision']:.4f}\n")
            f.write(f"  Recall      : {row['recall']:.4f}\n")
            f.write(f"  F1-score    : {row['f1']:.4f}\n")
            f.write(f"  Fraudes detectees : {int(row['tp']):,} / {n_frauds:,}\n")
            f.write(f"  Faux positifs     : {int(row['fp']):,}\n")
            f.write(f"  Alertes totales   : {int(row['alerts']):,}\n")
            f.write(f"  Benefice net      : ${row['net_benefit']:,.0f}\n")

    return report_path


def write_strategy_note(scenarios):
    if RECOMMENDED_SCENARIO not in scenarios:
        return None

    recommended = scenarios[RECOMMENDED_SCENARIO]
    best_profit = scenarios.get("best_profit")
    best_f1 = scenarios.get("best_f1")
    path = os.path.join(RESULTS_DIR, "base_threshold_business_strategy.txt")

    with open(path, "w", encoding="utf-8") as f:
        f.write("STRATEGIE DE SEUIL - BANQUE DIGITALE\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Choix retenu : seuil {recommended['threshold']:.2f}, scenario {format_scenario(RECOMMENDED_SCENARIO)}.\n\n")
        f.write("Positionnement business\n")
        f.write("-" * 50 + "\n")
        f.write("Le seuil retenu ne cherche pas seulement a maximiser une metrique ML.\n")
        f.write("Il traduit une politique de risque exploitable par une banque digitale :\n")
        f.write("detecter suffisamment de fraudes, maintenir une precision elevee, et\n")
        f.write("eviter une surcharge excessive des equipes d'investigation.\n\n")

        f.write("Resultats du seuil retenu\n")
        f.write("-" * 50 + "\n")
        f.write(f"Seuil       : {recommended['threshold']:.2f}\n")
        f.write(f"Precision   : {recommended['precision']:.4f}\n")
        f.write(f"Recall      : {recommended['recall']:.4f}\n")
        f.write(f"F1-score    : {recommended['f1']:.4f}\n")
        f.write(f"Faux positifs: {int(recommended['fp']):,}\n")
        f.write(f"Benefice net: ${recommended['net_benefit']:,.0f}\n\n")

        if best_profit is not None:
            benefit_gap = best_profit["net_benefit"] - recommended["net_benefit"]
            fp_reduction = best_profit["fp"] - recommended["fp"]
            f.write("Comparaison avec le seuil de benefice maximal\n")
            f.write("-" * 50 + "\n")
            f.write(f"Seuil profit max : {best_profit['threshold']:.2f}\n")
            f.write(f"Benefice perdu   : ${benefit_gap:,.0f}\n")
            f.write(f"Faux positifs evites : {int(fp_reduction):,}\n")
            f.write(f"Lecture : le seuil {recommended['threshold']:.2f} accepte un benefice net plus faible, mais il\n")
            f.write("reduit fortement la charge operationnelle et protege mieux l'experience client.\n\n")

        if best_f1 is not None:
            f.write("Comparaison avec le seuil meilleur F1\n")
            f.write("-" * 50 + "\n")
            f.write(f"Seuil meilleur F1 : {best_f1['threshold']:.2f}\n")
            f.write(f"F1 seuil retenu   : {recommended['f1']:.4f}\n")
            f.write(f"F1 meilleur F1    : {best_f1['f1']:.4f}\n")
            f.write("Lecture : la performance statistique reste quasiment au niveau du meilleur F1,\n")
            f.write("avec un positionnement metier plus oriente detection et benefice.\n\n")

        f.write("Formulation soutenance\n")
        f.write("-" * 50 + "\n")
        f.write(f"Nous retenons un seuil de {recommended['threshold']:.2f} car il maximise le F1-score\n")
        f.write("du modele final optimise tout en conservant une precision elevee et un recall fort.\n")
        f.write("Ce choix est coherent pour une banque digitale : il detecte davantage de fraudes\n")
        f.write("sans generer une charge excessive de faux positifs, et il garde un benefice net\n")
        f.write("metier eleve.\n")

    return path


def print_scenarios(y_true, y_proba, scenarios):
    print("=" * 82)
    print("  ADAPTATION DU SEUIL - MODELE BASE XGBOOST")
    print("=" * 82)
    print(f"  AUC-ROC: {roc_auc_score(y_true, y_proba):.4f}")
    print(f"  Average Precision: {average_precision_score(y_true, y_proba):.4f}\n")
    print(f"  {'Scenario':<28} {'Seuil':>7} {'Prec':>8} {'Recall':>8} {'F1':>8} {'FP':>8} {'Benefice':>12}")
    print("  " + "-" * 78)

    for name, row in scenarios.items():
        marker = " <- retenu" if name == RECOMMENDED_SCENARIO else ""
        print(
            f"  {format_scenario(name):<28} "
            f"{row['threshold']:>7.2f} "
            f"{row['precision']:>8.3f} "
            f"{row['recall']:>8.3f} "
            f"{row['f1']:>8.3f} "
            f"{int(row['fp']):>8,} "
            f"${row['net_benefit']:>11,.0f}"
            f"{marker}"
        )


def run():
    y_true, y_proba = load_predictions()
    analysis = analyze_thresholds(y_true, y_proba)
    scenarios = select_scenarios(analysis)
    figure_path = plot_thresholds(analysis, scenarios)
    report_path = write_report(y_true, y_proba, analysis, scenarios, figure_path)
    strategy_path = write_strategy_note(scenarios)

    print_scenarios(y_true, y_proba, scenarios)
    print("\nFichiers generes :")
    print(f"  Rapport : {report_path}")
    print(f"  Figure  : {figure_path}")
    if strategy_path is not None:
        print(f"  Strategie : {strategy_path}")


if __name__ == "__main__":
    run()
