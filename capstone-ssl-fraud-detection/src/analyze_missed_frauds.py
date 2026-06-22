"""
=================================================================
Analyse des fraudes non détectées
=================================================================
Les 536 fraudes avec un score < 0.40 passent sous le radar.
Ce script analyse leur profil pour comprendre POURQUOI
le modèle les rate et proposer des contre-mesures.

Usage :
    python src/analyze_missed_frauds.py
=================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS_DIR = os.path.join("..", "reports", "results")
FIGURES_DIR = os.path.join("..", "reports", "figures")
RAW_DIR = os.path.join("..", "data", "raw")
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

SEUIL = 0.40


def run():
    print("=" * 60)
    print("  ANALYSE DES FRAUDES NON DÉTECTÉES")
    print("=" * 60)

    # 1. Charger les prédictions
    print("\n1. Chargement des prédictions...")
    preds = np.load(os.path.join(RESULTS_DIR, "final_predictions.npz"))
    y_true = preds["y_true"]
    y_proba = preds["y_proba"]

    # 2. Charger les données brutes pour l'analyse
    print("2. Chargement des données brutes...")
    tx = pd.read_csv(os.path.join(RAW_DIR, "train_transaction.csv"))
    ident = pd.read_csv(os.path.join(RAW_DIR, "train_identity.csv"))
    df_raw = tx.merge(ident, on="TransactionID", how="left")

    # Récupérer le même split val (20% dernier)
    np.random.seed(42)
    from sklearn.model_selection import train_test_split
    _, val_idx = train_test_split(
        range(len(df_raw)), test_size=0.20, random_state=42,
        stratify=df_raw["isFraud"]
    )
    df_val = df_raw.iloc[val_idx].reset_index(drop=True)
    df_val["score"] = y_proba
    df_val["prediction"] = (y_proba >= SEUIL).astype(int)

    # 3. Séparer les groupes
    all_frauds = df_val[df_val["isFraud"] == 1]
    detected = all_frauds[all_frauds["score"] >= SEUIL]
    missed = all_frauds[all_frauds["score"] < SEUIL]
    legit = df_val[df_val["isFraud"] == 0]

    n_total = len(all_frauds)
    n_detected = len(detected)
    n_missed = len(missed)

    print(f"\n   Fraudes totales    : {n_total:,}")
    print(f"   Détectées (>={SEUIL}) : {n_detected:,} ({n_detected/n_total*100:.1f}%)")
    print(f"   Ratées (<{SEUIL})     : {n_missed:,} ({n_missed/n_total*100:.1f}%)")

    # 4. Comparer les profils
    print(f"\n{'='*60}")
    print("  PROFIL DES FRAUDES RATÉES vs DÉTECTÉES")
    print(f"{'='*60}")

    # Montant
    print(f"\n  MONTANT :")
    print(f"    Fraudes ratées   : médiane ${missed['TransactionAmt'].median():.0f}, moyenne ${missed['TransactionAmt'].mean():.0f}")
    print(f"    Fraudes détectées: médiane ${detected['TransactionAmt'].median():.0f}, moyenne ${detected['TransactionAmt'].mean():.0f}")
    print(f"    Légitimes        : médiane ${legit['TransactionAmt'].median():.0f}, moyenne ${legit['TransactionAmt'].mean():.0f}")

    amt_ratio = missed["TransactionAmt"].median() / max(detected["TransactionAmt"].median(), 1)
    if amt_ratio < 0.8:
        print(f"    --> Les fraudes ratées ont des montants PLUS PETITS (elles ressemblent à des transactions normales)")
    elif amt_ratio > 1.2:
        print(f"    --> Les fraudes ratées ont des montants PLUS GRANDS")
    else:
        print(f"    --> Montants similaires")

    # Produit
    print(f"\n  PRODUIT :")
    for group_name, group in [("Ratées", missed), ("Détectées", detected), ("Légitimes", legit)]:
        dist = group["ProductCD"].value_counts(normalize=True).head(5)
        top = ", ".join([f"{k}:{v*100:.0f}%" for k, v in dist.items()])
        print(f"    {group_name:12s}: {top}")

    # Carte
    print(f"\n  TYPE DE CARTE :")
    for group_name, group in [("Ratées", missed), ("Détectées", detected)]:
        if "card6" in group.columns:
            dist = group["card6"].value_counts(normalize=True)
            top = ", ".join([f"{k}:{v*100:.0f}%" for k, v in dist.head(3).items()])
            print(f"    {group_name:12s}: {top}")

    # Heure
    if "TransactionDT" in missed.columns:
        missed_hour = (missed["TransactionDT"] / 3600 % 24).astype(int)
        detected_hour = (detected["TransactionDT"] / 3600 % 24).astype(int)
        legit_hour = (legit["TransactionDT"] / 3600 % 24).astype(int)

        missed_peak = ((missed_hour >= 5) & (missed_hour <= 10)).mean() * 100
        detected_peak = ((detected_hour >= 5) & (detected_hour <= 10)).mean() * 100
        legit_peak = ((legit_hour >= 5) & (legit_hour <= 10)).mean() * 100

        print(f"\n  HEURE DE POINTE (5h-10h) :")
        print(f"    Ratées    : {missed_peak:.1f}% pendant le pic")
        print(f"    Détectées : {detected_peak:.1f}% pendant le pic")
        print(f"    Légitimes : {legit_peak:.1f}% pendant le pic")
        if missed_peak < detected_peak:
            print(f"    --> Les fraudes ratées agissent HORS pic (elles se cachent)")

    # Email
    print(f"\n  EMAIL (P_emaildomain) :")
    for group_name, group in [("Ratées", missed), ("Détectées", detected)]:
        dist = group["P_emaildomain"].value_counts(normalize=True).head(5)
        top = ", ".join([f"{k}:{v*100:.0f}%" for k, v in dist.items()])
        print(f"    {group_name:12s}: {top}")

    # Device
    print(f"\n  DEVICE :")
    for group_name, group in [("Ratées", missed), ("Détectées", detected)]:
        dist = group["DeviceType"].value_counts(normalize=True, dropna=False)
        top = ", ".join([f"{k}:{v*100:.0f}%" for k, v in dist.head(3).items()])
        print(f"    {group_name:12s}: {top}")

    # Valeurs manquantes
    missed_nan = missed.isnull().mean(axis=1).mean() * 100
    detected_nan = detected.isnull().mean(axis=1).mean() * 100
    legit_nan = legit.isnull().mean(axis=1).mean() * 100
    print(f"\n  VALEURS MANQUANTES (% moyen par transaction) :")
    print(f"    Ratées    : {missed_nan:.1f}%")
    print(f"    Détectées : {detected_nan:.1f}%")
    print(f"    Légitimes : {legit_nan:.1f}%")

    # 5. Score distribution
    print(f"\n  DISTRIBUTION DES SCORES :")
    print(f"    Ratées    : min={missed['score'].min():.3f}, max={missed['score'].max():.3f}, médiane={missed['score'].median():.3f}")
    print(f"    Détectées : min={detected['score'].min():.3f}, max={detected['score'].max():.3f}, médiane={detected['score'].median():.3f}")

    # Combien sont proches du seuil ?
    close = missed[missed["score"] >= SEUIL - 0.10]
    very_low = missed[missed["score"] < 0.10]
    print(f"\n    Proches du seuil (>{SEUIL-0.10:.2f}) : {len(close)} ({len(close)/n_missed*100:.1f}%)")
    print(f"    Très bas (<0.10)           : {len(very_low)} ({len(very_low)/n_missed*100:.1f}%)")

    # 6. Conclusion
    print(f"\n{'='*60}")
    print("  CONCLUSION")
    print(f"{'='*60}")
    print(f"\n  Sur {n_missed} fraudes ratées :")
    print(f"    - {len(close)} sont proches du seuil et pourraient être")
    print(f"      rattrapées en baissant le seuil à {SEUIL-0.10:.2f}")
    print(f"    - {len(very_low)} ont un score très bas (<0.10) :")
    print(f"      ce sont des fraudes qui ressemblent parfaitement")
    print(f"      à des transactions légitimes. Aucun modèle ML")
    print(f"      ne peut les détecter sans informations supplémentaires")
    print(f"      (appel client, vérification IP, historique comportemental).")
    print(f"\n  C'est la limite naturelle du machine learning :")
    print(f"  les fraudeurs sophistiqués imitent les comportements normaux.")
    print(f"  La détection complète nécessite une approche hybride")
    print(f"  ML + règles métier + vérification humaine.")

    # 7. Graphiques
    print(f"\n7. Graphiques...")

    # Distribution des scores des fraudes
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(detected["score"], bins=40, alpha=0.7, label=f"Detectees ({n_detected})", color="#4CAF50")
    axes[0].hist(missed["score"], bins=40, alpha=0.7, label=f"Ratees ({n_missed})", color="#E53935")
    axes[0].axvline(x=SEUIL, color="black", linestyle="--", linewidth=2, label=f"Seuil {SEUIL}")
    axes[0].set_xlabel("Score du modele", fontsize=11)
    axes[0].set_ylabel("Nombre de fraudes", fontsize=11)
    axes[0].set_title("Distribution des scores des fraudes", fontsize=12)
    axes[0].legend(fontsize=10)

    # Montant : ratées vs détectées
    data_box = pd.DataFrame({
        "Montant": pd.concat([
            missed["TransactionAmt"].clip(upper=500),
            detected["TransactionAmt"].clip(upper=500),
            legit["TransactionAmt"].clip(upper=500).sample(5000, random_state=42),
        ]),
        "Type": (["Fraude ratee"] * len(missed) +
                 ["Fraude detectee"] * len(detected) +
                 ["Legitime"] * 5000)
    })
    palette = {"Fraude ratee": "#E53935", "Fraude detectee": "#FF9800", "Legitime": "#2196F3"}
    sns.boxplot(data=data_box, x="Type", y="Montant", ax=axes[1], palette=palette)
    axes[1].set_title("Montant par type de transaction", fontsize=12)
    axes[1].set_ylabel("Montant ($)", fontsize=11)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "missed_frauds_analysis.png"), dpi=150)
    plt.close()
    print("   ✓ missed_frauds_analysis.png")

    # Sauvegarder le rapport
    with open(os.path.join(RESULTS_DIR, "missed_frauds_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"ANALYSE DES FRAUDES NON DETECTEES\n")
        f.write(f"Seuil : {SEUIL}\n")
        f.write(f"Fraudes ratees : {n_missed} / {n_total}\n\n")
        f.write(f"Montant median ratees    : ${missed['TransactionAmt'].median():.0f}\n")
        f.write(f"Montant median detectees : ${detected['TransactionAmt'].median():.0f}\n\n")
        f.write(f"Proches du seuil : {len(close)}\n")
        f.write(f"Score tres bas   : {len(very_low)}\n")
    print("   ✓ missed_frauds_report.txt")

    print(f"\n{'='*60}")
    print("ANALYSE TERMINEE")
    print(f"{'='*60}")


if __name__ == "__main__":
    run()
