"""
SSL-Guided Fraud Risk Scoring
--------------------------------

Ce script combine :
1) la probabilité de fraude produite par XGBoost,
2) le score d'anomalie issu du Self-Supervised Learning,
3) une heuristique métier simple,

puis optimise les poids alpha/beta/gamma et le seuil de décision
afin de maximiser le F1-score.

IMPORTANT :
Si le fichier d'entrée n'existe pas, le script le crée automatiquement
avec les bonnes colonnes, puis crée aussi un fichier .txt d'explication.

Commande depuis la racine du projet :
python src/ssl_guided_risk_scoring.py

Commande depuis le dossier src :
python .\ssl_guided_risk_scoring.py
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)


def resolve_project_root() -> Path:
    """
    Détecte automatiquement la racine du projet.
    Si le script est lancé depuis src, on remonte d'un niveau.
    """
    cwd = Path.cwd()

    if cwd.name.lower() == "src":
        return cwd.parent

    return cwd


def create_template_input(input_path: Path) -> None:
    """
    Crée automatiquement un fichier scoring_input.csv modèle
    si le fichier attendu n'existe pas encore.
    """
    input_path.parent.mkdir(parents=True, exist_ok=True)

    template_df = pd.DataFrame(
        {
            "y_true": [0, 1, 0],
            "xgb_proba": [0.12, 0.87, 0.34],
            "ssl_reconstruction_error": [0.02, 0.41, 0.10],
            "TransactionAmt": [45.20, 950.00, 130.50],
            "missing_ratio": [0.10, 0.65, 0.20],
        }
    )

    template_df.to_csv(input_path, index=False)

    txt_path = input_path.with_suffix(".txt")
    txt_path.write_text(
        """Fichier scoring_input.csv créé automatiquement.

Tu dois remplacer les exemples par tes vraies valeurs.

Colonnes obligatoires :
- y_true : vraie classe, 0 = non fraude, 1 = fraude
- xgb_proba : probabilité de fraude donnée par XGBoost
- ssl_reconstruction_error : erreur de reconstruction du modèle SSL

Colonnes optionnelles mais utiles :
- TransactionAmt : montant de la transaction
- missing_ratio : taux de valeurs manquantes sur la ligne

Exemple de création depuis ton notebook :

import pandas as pd
from pathlib import Path

Path("reports/results").mkdir(parents=True, exist_ok=True)

scoring_df = pd.DataFrame({
    "y_true": y_test,
    "xgb_proba": model.predict_proba(X_test_final)[:, 1],
    "ssl_reconstruction_error": reconstruction_error,
    "TransactionAmt": X_test_original["TransactionAmt"].values,
    "missing_ratio": X_test_original.isnull().mean(axis=1).values
})

scoring_df.to_csv("reports/results/scoring_input.csv", index=False)

Ensuite relance :
python src/ssl_guided_risk_scoring.py
""",
        encoding="utf-8",
    )

    print("\nLe fichier d'entrée n'existait pas, donc je l'ai créé automatiquement.")
    print(f"CSV créé ici : {input_path}")
    print(f"TXT d'explication créé ici : {txt_path}")
    print("\nRemplace les 3 lignes d'exemple par tes vraies prédictions, puis relance la commande.")
    print("Commande depuis la racine : python src/ssl_guided_risk_scoring.py")
    print("Commande depuis src        : python .\\ssl_guided_risk_scoring.py")


def minmax_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)

    min_val = np.nanmin(values)
    max_val = np.nanmax(values)

    if np.isclose(max_val, min_val):
        return np.zeros_like(values, dtype=float)

    return (values - min_val) / (max_val - min_val)


def build_ssl_anomaly_score(
    df: pd.DataFrame,
    ssl_score_col: str,
    ssl_error_col: str,
) -> np.ndarray:
    if ssl_score_col in df.columns:
        return minmax_normalize(df[ssl_score_col].values)

    if ssl_error_col in df.columns:
        return minmax_normalize(df[ssl_error_col].values)

    raise ValueError(
        f"Aucune colonne SSL trouvée. Ajoute '{ssl_score_col}' ou '{ssl_error_col}' dans le CSV."
    )


def build_heuristic_score(
    df: pd.DataFrame,
    amount_col: str,
    missing_col: str,
    heuristic_col: str,
) -> np.ndarray:
    if heuristic_col in df.columns:
        return minmax_normalize(df[heuristic_col].values)

    heuristic = np.zeros(len(df), dtype=float)
    used_rule_count = 0

    if amount_col in df.columns:
        amount_threshold = np.nanpercentile(df[amount_col].values, 95)
        heuristic += (df[amount_col].values > amount_threshold).astype(float)
        used_rule_count += 1

    if missing_col in df.columns:
        heuristic += (df[missing_col].values > 0.5).astype(float)
        used_rule_count += 1

    if used_rule_count == 0:
        return np.zeros(len(df), dtype=float)

    return heuristic / used_rule_count


def evaluate_scores(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    y_pred = (scores >= threshold).astype(int)

    result = {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    try:
        result["auc_roc"] = float(roc_auc_score(y_true, scores))
    except ValueError:
        result["auc_roc"] = None

    try:
        result["auc_pr"] = float(average_precision_score(y_true, scores))
    except ValueError:
        result["auc_pr"] = None

    return result


def optimize_weights_and_threshold(
    y_true: np.ndarray,
    xgb_proba: np.ndarray,
    ssl_anomaly_score: np.ndarray,
    heuristic_score: np.ndarray,
    weight_step: float,
    threshold_step: float,
) -> dict:
    best = {
        "f1": -1,
        "precision": None,
        "recall": None,
        "auc_roc": None,
        "auc_pr": None,
        "alpha": None,
        "beta": None,
        "gamma": None,
        "threshold": None,
        "confusion_matrix": None,
    }

    weights = np.arange(0, 1 + weight_step, weight_step)
    thresholds = np.arange(0.01, 0.99 + threshold_step, threshold_step)

    heuristic_is_available = not np.allclose(heuristic_score, 0)

    for alpha in weights:
        for beta in weights:
            gamma = 1 - alpha - beta

            if gamma < -1e-9:
                continue

            gamma = max(0, gamma)

            if not heuristic_is_available and gamma > 1e-9:
                continue

            if np.isclose(alpha + beta + gamma, 0):
                continue

            final_score = (
                alpha * xgb_proba
                + beta * ssl_anomaly_score
                + gamma * heuristic_score
            )

            for threshold in thresholds:
                metrics = evaluate_scores(y_true, final_score, threshold)

                if metrics["f1"] > best["f1"]:
                    best.update(
                        {
                            "f1": metrics["f1"],
                            "precision": metrics["precision"],
                            "recall": metrics["recall"],
                            "auc_roc": metrics["auc_roc"],
                            "auc_pr": metrics["auc_pr"],
                            "alpha": float(alpha),
                            "beta": float(beta),
                            "gamma": float(gamma),
                            "threshold": float(threshold),
                            "confusion_matrix": metrics["confusion_matrix"],
                        }
                    )

    return best


def main() -> None:
    project_root = resolve_project_root()
    default_input = project_root / "reports" / "results" / "scoring_input.csv"
    default_output = project_root / "reports" / "results" / "ssl_guided_scores.csv"

    parser = argparse.ArgumentParser(
        description="SSL-Guided Fraud Risk Scoring with automatic input template creation."
    )

    parser.add_argument("--input", default=str(default_input), help="Chemin du CSV d'entrée.")
    parser.add_argument("--output", default=str(default_output), help="Chemin du CSV de sortie.")

    parser.add_argument("--label-col", default="y_true")
    parser.add_argument("--proba-col", default="xgb_proba")

    parser.add_argument("--ssl-score-col", default="ssl_anomaly_score")
    parser.add_argument("--ssl-error-col", default="ssl_reconstruction_error")

    parser.add_argument("--amount-col", default="TransactionAmt")
    parser.add_argument("--missing-col", default="missing_ratio")
    parser.add_argument("--heuristic-col", default="heuristic_score")

    parser.add_argument("--weight-step", type=float, default=0.05)
    parser.add_argument("--threshold-step", type=float, default=0.01)

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        create_template_input(input_path)
        return

    df = pd.read_csv(input_path)

    required_cols = [args.label_col, args.proba_col]
    missing_required = [col for col in required_cols if col not in df.columns]

    if missing_required:
        raise ValueError(f"Colonnes obligatoires manquantes : {missing_required}")

    y_true = df[args.label_col].astype(int).values
    xgb_proba = minmax_normalize(df[args.proba_col].values)

    ssl_anomaly_score = build_ssl_anomaly_score(
        df,
        ssl_score_col=args.ssl_score_col,
        ssl_error_col=args.ssl_error_col,
    )

    heuristic_score = build_heuristic_score(
        df,
        amount_col=args.amount_col,
        missing_col=args.missing_col,
        heuristic_col=args.heuristic_col,
    )

    best = optimize_weights_and_threshold(
        y_true=y_true,
        xgb_proba=xgb_proba,
        ssl_anomaly_score=ssl_anomaly_score,
        heuristic_score=heuristic_score,
        weight_step=args.weight_step,
        threshold_step=args.threshold_step,
    )

    final_score = (
        best["alpha"] * xgb_proba
        + best["beta"] * ssl_anomaly_score
        + best["gamma"] * heuristic_score
    )

    y_pred = (final_score >= best["threshold"]).astype(int)

    df["xgb_proba_normalized"] = xgb_proba
    df["ssl_anomaly_score_final"] = ssl_anomaly_score
    df["heuristic_score_final"] = heuristic_score
    df["final_fraud_score"] = final_score
    df["y_pred"] = y_pred

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    summary_path = output_path.with_suffix(".summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(best, f, indent=4)

    print("\n=== Best SSL-Guided Risk Scoring Result ===")
    print(f"Alpha XGBoost     : {best['alpha']}")
    print(f"Beta SSL anomaly  : {best['beta']}")
    print(f"Gamma heuristic   : {best['gamma']}")
    print(f"Best threshold    : {best['threshold']}")
    print(f"Precision         : {best['precision']:.4f}")
    print(f"Recall            : {best['recall']:.4f}")
    print(f"F1-score          : {best['f1']:.4f}")

    if best["auc_roc"] is not None:
        print(f"AUC-ROC           : {best['auc_roc']:.4f}")

    if best["auc_pr"] is not None:
        print(f"AUC-PR            : {best['auc_pr']:.4f}")

    print("\nConfusion matrix:")
    print(np.array(best["confusion_matrix"]))

    print("\nClassification report:")
    print(classification_report(y_true, y_pred, zero_division=0))

    print(f"\nFichier de prédictions sauvegardé : {output_path}")
    print(f"Résumé sauvegardé : {summary_path}")


if __name__ == "__main__":
    main()
