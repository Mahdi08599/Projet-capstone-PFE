"""
=================================================================
Analyse Métier — KPIs de Détection de Fraude Bancaire
=================================================================
Ce script répond aux vraies questions business d'une banque :

  1. Quel est le coût financier de la fraude ?
  2. Quels produits/canaux sont les plus ciblés ?
  3. Quel est le profil type d'une transaction frauduleuse ?
  4. À quels moments les fraudes sont-elles concentrées ?
  5. Quels devices/emails sont des signaux de risque ?
  6. Quel serait l'impact métier de notre modèle SSL ?

Usage :
    python src/business_analysis.py

Sortie :
    reports/figures/business_*.png
    reports/results/kpi_report.txt
=================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

RAW_DIR = os.path.join("..", "data", "raw")
FIGURES_DIR = os.path.join("..", "reports", "figures")
RESULTS_DIR = os.path.join("..", "reports", "results")
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

sns.set_style("whitegrid")


def load_raw_data():
    """Charge les données brutes pour l'analyse métier."""
    print("Chargement des données brutes...")
    tx = pd.read_csv(os.path.join(RAW_DIR, "train_transaction.csv"))
    identity = pd.read_csv(os.path.join(RAW_DIR, "train_identity.csv"))
    df = tx.merge(identity, on="TransactionID", how="left")
    print(f"  {len(df):,} transactions chargées")
    return df


def kpi_cout_financier(df, report):
    """KPI 1 : Impact financier de la fraude."""
    print("\n─── KPI 1 : COÛT FINANCIER DE LA FRAUDE ───")

    fraud = df[df["isFraud"] == 1]
    legit = df[df["isFraud"] == 0]

    total_amount = df["TransactionAmt"].sum()
    fraud_amount = fraud["TransactionAmt"].sum()
    legit_amount = legit["TransactionAmt"].sum()

    avg_fraud = fraud["TransactionAmt"].mean()
    avg_legit = legit["TransactionAmt"].mean()
    median_fraud = fraud["TransactionAmt"].median()
    median_legit = legit["TransactionAmt"].median()

    report.append("=" * 55)
    report.append("KPI 1 : IMPACT FINANCIER")
    report.append("=" * 55)
    report.append(f"Volume total des transactions : ${total_amount:,.0f}")
    report.append(f"Volume frauduleux             : ${fraud_amount:,.0f} ({fraud_amount/total_amount*100:.2f}%)")
    report.append(f"Nombre de fraudes             : {len(fraud):,} / {len(df):,} ({len(fraud)/len(df)*100:.2f}%)")
    report.append(f"Montant moyen fraude          : ${avg_fraud:,.2f}")
    report.append(f"Montant moyen légitime        : ${avg_legit:,.2f}")
    report.append(f"Montant médian fraude          : ${median_fraud:,.2f}")
    report.append(f"Montant médian légitime        : ${median_legit:,.2f}")
    report.append(f"Ratio montant moyen fraude/légitime : {avg_fraud/avg_legit:.1f}x")

    print(f"  Volume frauduleux : ${fraud_amount:,.0f} ({fraud_amount/total_amount*100:.2f}%)")
    print(f"  Montant moyen fraude : ${avg_fraud:.2f} vs légitime : ${avg_legit:.2f}")

    # Graphique : distribution des montants
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(legit["TransactionAmt"].clip(upper=500), bins=80, alpha=0.7, label="Légitime", color="#2196F3")
    axes[0].hist(fraud["TransactionAmt"].clip(upper=500), bins=80, alpha=0.7, label="Fraude", color="#E53935")
    axes[0].set_xlabel("Montant ($)", fontsize=11)
    axes[0].set_ylabel("Nombre de transactions", fontsize=11)
    axes[0].set_title("Distribution des montants (< $500)", fontsize=12)
    axes[0].legend(fontsize=10)

    data_box = pd.DataFrame({
        "Montant": pd.concat([legit["TransactionAmt"].clip(upper=1000),
                              fraud["TransactionAmt"].clip(upper=1000)]),
        "Type": ["Légitime"] * len(legit) + ["Fraude"] * len(fraud)
    })
    sns.boxplot(data=data_box, x="Type", y="Montant", ax=axes[1],
                palette={"Légitime": "#2196F3", "Fraude": "#E53935"})
    axes[1].set_title("Comparaison des montants", fontsize=12)
    axes[1].set_ylabel("Montant ($)", fontsize=11)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "business_montants.png"), dpi=150)
    plt.close()
    print("  ✓ business_montants.png")


def kpi_produits_cibles(df, report):
    """KPI 2 : Quels produits sont les plus ciblés par la fraude ?"""
    print("\n─── KPI 2 : PRODUITS CIBLÉS ───")

    product_stats = df.groupby("ProductCD").agg(
        total=("isFraud", "count"),
        frauds=("isFraud", "sum"),
        montant_moyen=("TransactionAmt", "mean"),
        montant_total=("TransactionAmt", "sum"),
    ).reset_index()
    product_stats["taux_fraude"] = (product_stats["frauds"] / product_stats["total"] * 100)
    product_stats["montant_fraude"] = df[df["isFraud"]==1].groupby("ProductCD")["TransactionAmt"].sum().values
    product_stats = product_stats.sort_values("taux_fraude", ascending=False)

    report.append("\n" + "=" * 55)
    report.append("KPI 2 : TAUX DE FRAUDE PAR PRODUIT")
    report.append("=" * 55)
    for _, row in product_stats.iterrows():
        report.append(f"  {row['ProductCD']} : {row['taux_fraude']:.2f}% "
                      f"({row['frauds']:,.0f} fraudes / {row['total']:,} tx) "
                      f"| Montant moyen : ${row['montant_moyen']:.0f}")

    print(product_stats[["ProductCD", "total", "frauds", "taux_fraude"]].to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].bar(product_stats["ProductCD"], product_stats["taux_fraude"],
                color=["#E53935" if t > 5 else "#FF9800" if t > 3 else "#4CAF50"
                       for t in product_stats["taux_fraude"]])
    axes[0].set_xlabel("Code produit", fontsize=11)
    axes[0].set_ylabel("Taux de fraude (%)", fontsize=11)
    axes[0].set_title("Taux de fraude par produit", fontsize=12)
    for i, (_, row) in enumerate(product_stats.iterrows()):
        axes[0].text(i, row["taux_fraude"] + 0.2, f"{row['taux_fraude']:.1f}%",
                     ha="center", fontsize=10, fontweight="bold")

    axes[1].bar(product_stats["ProductCD"], product_stats["frauds"], color="#E53935", alpha=0.8)
    axes[1].set_xlabel("Code produit", fontsize=11)
    axes[1].set_ylabel("Nombre de fraudes", fontsize=11)
    axes[1].set_title("Volume de fraudes par produit", fontsize=12)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "business_produits.png"), dpi=150)
    plt.close()
    print("  ✓ business_produits.png")


def kpi_cartes(df, report):
    """KPI 3 : Quel type de carte est le plus à risque ?"""
    print("\n─── KPI 3 : RISQUE PAR TYPE DE CARTE ───")

    report.append("\n" + "=" * 55)
    report.append("KPI 3 : RISQUE PAR TYPE DE CARTE")
    report.append("=" * 55)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, col, title in [(axes[0], "card4", "Réseau de carte"),
                           (axes[1], "card6", "Type de carte")]:
        stats = df.groupby(col).agg(
            total=("isFraud", "count"),
            frauds=("isFraud", "sum"),
        ).reset_index()
        stats["taux"] = stats["frauds"] / stats["total"] * 100
        stats = stats.sort_values("taux", ascending=True)

        colors = ["#E53935" if t > 4 else "#FF9800" if t > 3 else "#4CAF50" for t in stats["taux"]]
        ax.barh(stats[col].astype(str), stats["taux"], color=colors)
        ax.set_xlabel("Taux de fraude (%)", fontsize=11)
        ax.set_title(title, fontsize=12)
        for i, (_, row) in enumerate(stats.iterrows()):
            ax.text(row["taux"] + 0.1, i, f"{row['taux']:.2f}%  ({row['frauds']:,.0f})",
                    va="center", fontsize=9)

        for _, row in stats.iterrows():
            report.append(f"  {col}={row[col]} : {row['taux']:.2f}% "
                          f"({row['frauds']:,.0f} / {row['total']:,})")

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "business_cartes.png"), dpi=150)
    plt.close()
    print("  ✓ business_cartes.png")


def kpi_temporel(df, report):
    """KPI 4 : Quand les fraudes se produisent-elles ?"""
    print("\n─── KPI 4 : ANALYSE TEMPORELLE ───")

    # TransactionDT = secondes depuis un point de référence
    df["hour"] = (df["TransactionDT"] / 3600 % 24).astype(int)
    df["day"] = (df["TransactionDT"] / 86400).astype(int)

    hourly = df.groupby("hour").agg(
        total=("isFraud", "count"),
        frauds=("isFraud", "sum"),
    ).reset_index()
    hourly["taux"] = hourly["frauds"] / hourly["total"] * 100

    peak_hour = hourly.loc[hourly["taux"].idxmax(), "hour"]
    low_hour = hourly.loc[hourly["taux"].idxmin(), "hour"]

    report.append("\n" + "=" * 55)
    report.append("KPI 4 : PATTERNS TEMPORELS")
    report.append("=" * 55)
    report.append(f"  Heure la plus risquée : {peak_hour}h ({hourly.loc[hourly['taux'].idxmax(), 'taux']:.2f}%)")
    report.append(f"  Heure la moins risquée : {low_hour}h ({hourly.loc[hourly['taux'].idxmin(), 'taux']:.2f}%)")

    print(f"  Heure pic fraude : {peak_hour}h | Heure calme : {low_hour}h")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(hourly["hour"], hourly["taux"],
                color=["#E53935" if t > 4.5 else "#FF9800" if t > 3.5 else "#4CAF50"
                       for t in hourly["taux"]])
    axes[0].set_xlabel("Heure de la journée", fontsize=11)
    axes[0].set_ylabel("Taux de fraude (%)", fontsize=11)
    axes[0].set_title("Taux de fraude par heure", fontsize=12)
    axes[0].set_xticks(range(0, 24))

    axes[1].plot(hourly["hour"], hourly["total"], "o-", label="Total", color="#2196F3", linewidth=2)
    ax2 = axes[1].twinx()
    ax2.plot(hourly["hour"], hourly["frauds"], "s-", label="Fraudes", color="#E53935", linewidth=2)
    axes[1].set_xlabel("Heure", fontsize=11)
    axes[1].set_ylabel("Total transactions", fontsize=11, color="#2196F3")
    ax2.set_ylabel("Nombre de fraudes", fontsize=11, color="#E53935")
    axes[1].set_title("Volume par heure", fontsize=12)
    axes[1].set_xticks(range(0, 24))

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "business_temporel.png"), dpi=150)
    plt.close()
    print("  ✓ business_temporel.png")


def kpi_emails(df, report):
    """KPI 5 : Quels domaines email sont des signaux de risque ?"""
    print("\n─── KPI 5 : DOMAINES EMAIL À RISQUE ───")

    email_stats = df.groupby("P_emaildomain").agg(
        total=("isFraud", "count"),
        frauds=("isFraud", "sum"),
    ).reset_index()
    email_stats["taux"] = email_stats["frauds"] / email_stats["total"] * 100
    email_stats = email_stats[email_stats["total"] >= 500].sort_values("taux", ascending=False)

    report.append("\n" + "=" * 55)
    report.append("KPI 5 : DOMAINES EMAIL LES PLUS RISQUÉS (min 500 tx)")
    report.append("=" * 55)

    top_risk = email_stats.head(10)
    for _, row in top_risk.iterrows():
        report.append(f"  {row['P_emaildomain']:25s} : {row['taux']:.2f}% "
                      f"({row['frauds']:,.0f} / {row['total']:,})")

    print(top_risk[["P_emaildomain", "total", "frauds", "taux"]].head(10).to_string(index=False))

    fig, ax = plt.subplots(figsize=(10, 6))
    top15 = email_stats.head(15)
    colors = ["#E53935" if t > 6 else "#FF9800" if t > 4 else "#4CAF50" for t in top15["taux"]]
    ax.barh(top15["P_emaildomain"], top15["taux"], color=colors)
    ax.set_xlabel("Taux de fraude (%)", fontsize=11)
    ax.set_title("Top 15 domaines email par taux de fraude", fontsize=12)
    ax.invert_yaxis()
    for i, (_, row) in enumerate(top15.iterrows()):
        ax.text(row["taux"] + 0.1, i, f"{row['taux']:.1f}%", va="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "business_emails.png"), dpi=150)
    plt.close()
    print("  ✓ business_emails.png")


def kpi_devices(df, report):
    """KPI 6 : Mobile vs Desktop — quel canal est le plus risqué ?"""
    print("\n─── KPI 6 : RISQUE PAR DEVICE ───")

    device_stats = df.groupby("DeviceType").agg(
        total=("isFraud", "count"),
        frauds=("isFraud", "sum"),
    ).reset_index().dropna()
    device_stats["taux"] = device_stats["frauds"] / device_stats["total"] * 100

    report.append("\n" + "=" * 55)
    report.append("KPI 6 : RISQUE PAR DEVICE")
    report.append("=" * 55)
    for _, row in device_stats.iterrows():
        report.append(f"  {row['DeviceType']:10s} : {row['taux']:.2f}% "
                      f"({row['frauds']:,.0f} / {row['total']:,})")

    # Transactions sans identité (pas de device info)
    no_device = df["DeviceType"].isna().sum()
    no_device_fraud = df[df["DeviceType"].isna()]["isFraud"].mean() * 100
    report.append(f"  {'Sans info':10s} : {no_device_fraud:.2f}% ({no_device:,} tx)")

    print(device_stats.to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 5))
    all_types = list(device_stats["DeviceType"]) + ["Sans info"]
    all_taux = list(device_stats["taux"]) + [no_device_fraud]
    colors = ["#2196F3", "#FF9800", "#888888"]
    ax.bar(all_types, all_taux, color=colors[:len(all_types)])
    ax.set_ylabel("Taux de fraude (%)", fontsize=11)
    ax.set_title("Taux de fraude par type de device", fontsize=12)
    for i, t in enumerate(all_taux):
        ax.text(i, t + 0.1, f"{t:.2f}%", ha="center", fontsize=11, fontweight="bold")

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "business_devices.png"), dpi=150)
    plt.close()
    print("  ✓ business_devices.png")


def kpi_impact_modele(df, report):
    """KPI 7 : Quel serait l'impact métier du modèle SSL ?"""
    print("\n─── KPI 7 : IMPACT MÉTIER DU MODÈLE ───")

    fraud = df[df["isFraud"] == 1]
    total_fraud_amount = fraud["TransactionAmt"].sum()
    n_frauds = len(fraud)

    # Résultats V2
    recall_v2 = 0.4319
    precision_v2 = 0.5047
    n_detected = int(n_frauds * recall_v2)
    amount_saved = total_fraud_amount * recall_v2
    n_false_alerts = int(n_detected / precision_v2) - n_detected

    # Coût estimé d'une investigation manuelle (industrie bancaire)
    cost_per_investigation = 15  # dollars
    cost_investigations = (n_detected + n_false_alerts) * cost_per_investigation

    report.append("\n" + "=" * 55)
    report.append("KPI 7 : IMPACT MÉTIER DU MODÈLE SSL")
    report.append("=" * 55)
    report.append(f"  Fraudes dans le dataset       : {n_frauds:,}")
    report.append(f"  Montant total frauduleux      : ${total_fraud_amount:,.0f}")
    report.append(f"")
    report.append(f"  Avec notre modèle SSL (recall={recall_v2:.0%}) :")
    report.append(f"    Fraudes détectées            : {n_detected:,} / {n_frauds:,}")
    report.append(f"    Montant sauvé (estimé)       : ${amount_saved:,.0f}")
    report.append(f"    Fausses alertes              : {n_false_alerts:,}")
    report.append(f"    Coût d'investigation         : ${cost_investigations:,.0f}")
    report.append(f"    Bénéfice net estimé          : ${amount_saved - cost_investigations:,.0f}")
    report.append(f"")
    report.append(f"  Sans modèle (tout manuel) :")
    report.append(f"    Coût si on vérifie tout      : ${len(df) * cost_per_investigation:,.0f}")
    report.append(f"    Coût si on ne vérifie rien   : ${total_fraud_amount:,.0f} de pertes")

    print(f"  Fraudes détectées : {n_detected:,} / {n_frauds:,}")
    print(f"  Montant sauvé    : ${amount_saved:,.0f}")
    print(f"  Bénéfice net     : ${amount_saved - cost_investigations:,.0f}")


def kpi_desequilibre(df, report):
    """KPI 8 : Visualisation du déséquilibre — la raison du SSL."""
    print("\n─── KPI 8 : DÉSÉQUILIBRE DES CLASSES ───")

    n_fraud = df["isFraud"].sum()
    n_legit = len(df) - n_fraud

    report.append("\n" + "=" * 55)
    report.append("KPI 8 : DÉSÉQUILIBRE (JUSTIFICATION DU SSL)")
    report.append("=" * 55)
    report.append(f"  Légitimes  : {n_legit:,} ({n_legit/len(df)*100:.2f}%)")
    report.append(f"  Fraudes    : {n_fraud:,} ({n_fraud/len(df)*100:.2f}%)")
    report.append(f"  Ratio      : 1 fraude pour {n_legit//n_fraud} transactions légitimes")
    report.append(f"")
    report.append(f"  C'est pourquoi le self-supervised learning est pertinent :")
    report.append(f"  on peut utiliser les {n_legit:,} transactions non labellisées")
    report.append(f"  pour apprendre la structure normale des données.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].bar(["Légitimes", "Fraudes"], [n_legit, n_fraud],
                color=["#2196F3", "#E53935"])
    axes[0].set_ylabel("Nombre de transactions", fontsize=11)
    axes[0].set_title("Distribution des classes", fontsize=12)
    axes[0].text(0, n_legit + 5000, f"{n_legit:,}\n({n_legit/len(df)*100:.1f}%)",
                 ha="center", fontsize=11, fontweight="bold")
    axes[0].text(1, n_fraud + 5000, f"{n_fraud:,}\n({n_fraud/len(df)*100:.1f}%)",
                 ha="center", fontsize=11, fontweight="bold", color="#E53935")

    axes[1].pie([n_legit, n_fraud], labels=["Légitimes", "Fraudes"],
                colors=["#2196F3", "#E53935"], autopct="%1.1f%%",
                startangle=90, textprops={"fontsize": 12})
    axes[1].set_title("Proportion fraude / légitime", fontsize=12)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "business_desequilibre.png"), dpi=150)
    plt.close()
    print("  ✓ business_desequilibre.png")


def run():
    print("=" * 60)
    print("ANALYSE MÉTIER — KPIs FRAUDE BANCAIRE")
    print("=" * 60)

    df = load_raw_data()
    report = []

    kpi_cout_financier(df, report)
    kpi_produits_cibles(df, report)
    kpi_cartes(df, report)
    kpi_temporel(df, report)
    kpi_emails(df, report)
    kpi_devices(df, report)
    kpi_desequilibre(df, report)
    kpi_impact_modele(df, report)

    # Sauvegarder le rapport
    report_path = os.path.join(RESULTS_DIR, "kpi_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("RAPPORT KPIs — FRAUDE BANCAIRE DIGITALE\n")
        f.write(f"Dataset : IEEE-CIS Fraud Detection\n")
        f.write(f"Transactions : {len(df):,}\n\n")
        for line in report:
            f.write(line + "\n")

    print(f"\n{'=' * 60}")
    print(f"Rapport sauvé   : {report_path}")
    print(f"Graphiques dans : {FIGURES_DIR}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    run()
