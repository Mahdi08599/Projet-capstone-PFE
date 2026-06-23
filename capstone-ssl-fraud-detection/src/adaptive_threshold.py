"""
=================================================================
Seuils adaptatifs par segment de risque
=================================================================
Heuristique : au lieu d'un seuil global, chaque transaction
a un seuil ajusté selon son profil de risque.

  Produit C (11.7% fraude) → seuil bas (0.50) → on attrape plus
  Produit W (2.0% fraude)  → seuil haut (0.85) → on évite le bruit

Le modèle reste le même, on change juste la règle de décision.

Usage :
    python src/adaptive_threshold.py
=================================================================
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

RESULTS_DIR = os.path.join("..", "reports", "results")
RAW_DIR = os.path.join("..", "data", "raw")


def run():
    print("=" * 65)
    print("  SEUILS ADAPTATIFS PAR SEGMENT DE RISQUE")
    print("=" * 65)

    # 1. Charger prédictions + données brutes
    print("\n1. Chargement...")
    data = np.load(os.path.join(RESULTS_DIR, "final_predictions.npz"))
    y_true = data["y_true"]
    y_proba = data["y_proba"]

    tx = pd.read_csv(os.path.join(RAW_DIR, "train_transaction.csv"))
    idf = pd.read_csv(os.path.join(RAW_DIR, "train_identity.csv"))
    df = tx.merge(idf, on="TransactionID", how="left")

    _, val_idx = train_test_split(
        range(len(df)), test_size=0.20, random_state=42,
        stratify=df["isFraud"]
    )
    df = df.iloc[val_idx].reset_index(drop=True)

    # Ajouter les scores
    df["score"] = y_proba
    df["y_true"] = y_true
    df["hour"] = (df["TransactionDT"] / 3600 % 24).astype(int)

    # 2. Définir les segments et leurs seuils
    print("\n2. Définition des segments de risque...\n")

    # Chaque segment a un seuil adapté à son taux de fraude réel
    def get_adaptive_threshold(row):
        score = row["score"]
        seuil = 0.78  # seuil par défaut (équilibré)

        # Produit à haut risque → seuil plus bas
        if row["ProductCD"] == "C":
            seuil -= 0.20  # 11.7% de fraude → on est plus agressif
        elif row["ProductCD"] == "S":
            seuil -= 0.10  # 5.9%
        elif row["ProductCD"] == "H":
            seuil -= 0.08  # 4.8%

        # Carte de crédit → plus risqué
        if row.get("card6") == "credit":
            seuil -= 0.08  # 6.7% vs 2.4% débit

        # Réseau Discover → plus risqué
        if row.get("card4") == "discover":
            seuil -= 0.05  # 7.7%

        # Heure de pointe fraude (5h-10h)
        if 5 <= row["hour"] <= 10:
            seuil -= 0.10  # 7.8% vs 3.3%

        # Mobile → plus risqué
        if row.get("DeviceType") == "mobile":
            seuil -= 0.05  # 10.2% vs 6.5%

        # Email à risque
        email = row.get("P_emaildomain", "")
        if email == "mail.com":
            seuil -= 0.15  # 19%
        elif email == "outlook.com":
            seuil -= 0.08  # 9.5%
        elif email in ["hotmail.com", "live.com.mx"]:
            seuil -= 0.05  # 5%+

        # Produit W → bas risque, seuil plus haut
        if row["ProductCD"] == "W":
            seuil += 0.05  # 2.0%

        # Carte débit → moins risqué
        if row.get("card6") == "debit":
            seuil += 0.05

        # Borner le seuil entre 0.30 et 0.90
        seuil = max(0.30, min(0.90, seuil))

        return seuil

    print("   Calcul des seuils adaptatifs...")
    df["adaptive_threshold"] = df.apply(get_adaptive_threshold, axis=1)
    df["adaptive_pred"] = (df["score"] >= df["adaptive_threshold"]).astype(int)

    # 3. Résultats
    y_pred_adaptive = df["adaptive_pred"].values
    y_pred_fixed = (y_proba >= 0.78).astype(int)

    tp_a = ((y_pred_adaptive == 1) & (y_true == 1)).sum()
    fp_a = ((y_pred_adaptive == 1) & (y_true == 0)).sum()
    fn_a = ((y_pred_adaptive == 0) & (y_true == 1)).sum()
    prec_a = tp_a / (tp_a + fp_a)
    rec_a = tp_a / (tp_a + fn_a)
    f1_a = 2 * prec_a * rec_a / (prec_a + rec_a)

    tp_f = ((y_pred_fixed == 1) & (y_true == 1)).sum()
    fp_f = ((y_pred_fixed == 1) & (y_true == 0)).sum()
    fn_f = ((y_pred_fixed == 0) & (y_true == 1)).sum()
    prec_f = tp_f / (tp_f + fp_f)
    rec_f = tp_f / (tp_f + fn_f)
    f1_f = 2 * prec_f * rec_f / (prec_f + rec_f)

    print(f"\n{'='*65}")
    print(f"  {'Métrique':<25} {'Seuil fixe 0.78':>15} {'Seuil adaptatif':>15} {'Gain':>10}")
    print(f"  {'-'*60}")
    print(f"  {'Fraudes détectées':<25} {tp_f:>15,} {tp_a:>15,} {tp_a-tp_f:>+10,}")
    print(f"  {'Fraudes ratées':<25} {fn_f:>15,} {fn_a:>15,} {fn_a-fn_f:>+10,}")
    print(f"  {'Fausses alertes':<25} {fp_f:>15,} {fp_a:>15,} {fp_a-fp_f:>+10,}")
    print(f"  {'Precision':<25} {prec_f:>15.3f} {prec_a:>15.3f} {prec_a-prec_f:>+10.3f}")
    print(f"  {'Recall':<25} {rec_f:>15.3f} {rec_a:>15.3f} {rec_a-rec_f:>+10.3f}")
    print(f"  {'F1-score':<25} {f1_f:>15.3f} {f1_a:>15.3f} {f1_a-f1_f:>+10.3f}")
    print(f"{'='*65}")

    # 4. Détail par segment
    print(f"\n  DÉTAIL PAR PRODUIT :")
    for prod in ["C", "W", "H", "R", "S"]:
        mask = df["ProductCD"] == prod
        if mask.sum() == 0:
            continue
        sub_true = y_true[mask]
        sub_pred = y_pred_adaptive[mask]
        sub_fixed = y_pred_fixed[mask]
        n_fraud = sub_true.sum()
        if n_fraud == 0:
            continue

        rec_adapt = ((sub_pred == 1) & (sub_true == 1)).sum() / n_fraud
        rec_fixed = ((sub_fixed == 1) & (sub_true == 1)).sum() / n_fraud
        avg_seuil = df.loc[mask, "adaptive_threshold"].mean()

        print(f"    Produit {prod} : seuil moyen={avg_seuil:.2f} | "
              f"Recall fixe={rec_fixed:.3f} | Recall adaptatif={rec_adapt:.3f} | "
              f"Gain={rec_adapt-rec_fixed:+.3f}")

    # 5. Distribution des seuils
    print(f"\n  DISTRIBUTION DES SEUILS ADAPTATIFS :")
    print(f"    Min    : {df['adaptive_threshold'].min():.2f}")
    print(f"    Q25    : {df['adaptive_threshold'].quantile(0.25):.2f}")
    print(f"    Médiane: {df['adaptive_threshold'].median():.2f}")
    print(f"    Q75    : {df['adaptive_threshold'].quantile(0.75):.2f}")
    print(f"    Max    : {df['adaptive_threshold'].max():.2f}")

    # 6. Impact financier
    saved_a = tp_a * 149
    cost_a = (tp_a + fp_a) * 15
    net_a = saved_a - cost_a
    saved_f = tp_f * 149
    cost_f = (tp_f + fp_f) * 15
    net_f = saved_f - cost_f

    print(f"\n  IMPACT FINANCIER :")
    print(f"    Seuil fixe     : bénéfice net = ${net_f:,.0f}")
    print(f"    Seuil adaptatif: bénéfice net = ${net_a:,.0f}")
    print(f"    Gain           : ${net_a-net_f:+,.0f}")

    print(f"\n{'='*65}")
    print("ANALYSE TERMINÉE")
    print(f"{'='*65}")


if __name__ == "__main__":
    run()
