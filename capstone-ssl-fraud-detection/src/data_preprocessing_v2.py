"""
=================================================================
Data Preprocessing V2 — Version améliorée
=================================================================
Corrections par rapport à la V1 :

  1. Seuil NaN relevé de 70% → 95% (on garde plus de colonnes)
  2. Indicateurs de valeurs manquantes (colonnes _missing)
  3. Frequency encoding au lieu de LabelEncoder
  4. Log-transformation du montant (TransactionAmt)
  5. Features temporelles extraites de TransactionDT

Logique adaptative :
  V1 → 208 colonnes supprimées, LabelEncoder naïf
  V2 → On conserve l'information, on enrichit les features

Usage :
    python src/data_preprocessing_v2.py
=================================================================
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

RAW_DIR = os.path.join("..", "data", "raw")
PROCESSED_DIR = os.path.join("..", "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

MISSING_THRESHOLD = 0.95   # V1 = 0.70, V2 = 0.95
MISSING_INDICATOR_THRESHOLD = 0.01  # colonnes avec >1% de NaN → indicateur
SEED = 42


def extract(raw_dir):
    """Phase 1 — Extract : identique à V1."""
    print("=" * 60)
    print("PHASE 1 — EXTRACT")
    print("=" * 60)

    t0 = time.time()
    train_tx = pd.read_csv(os.path.join(raw_dir, "train_transaction.csv"))
    train_id = pd.read_csv(os.path.join(raw_dir, "train_identity.csv"))

    df = train_tx.merge(train_id, on="TransactionID", how="left")
    print(f"  {df.shape[0]:,} lignes × {df.shape[1]} colonnes")
    print(f"  ✓ Extract en {time.time()-t0:.1f}s\n")
    return df


def transform_v2(df):
    """Phase 2 — Transform V2 : preprocessing amélioré."""
    print("=" * 60)
    print("PHASE 2 — TRANSFORM V2")
    print("=" * 60)

    t0 = time.time()
    report = []

    # ── 2.1 Suppression des colonnes quasi-vides (>95% NaN) ──
    print(f"  2.1 Suppression colonnes >{MISSING_THRESHOLD*100:.0f}% NaN...")
    missing_pct = df.isnull().mean()
    cols_to_drop = missing_pct[missing_pct > MISSING_THRESHOLD].index.tolist()
    cols_to_drop = [c for c in cols_to_drop if c not in ["TransactionID", "isFraud"]]

    n_before = df.shape[1]
    df = df.drop(columns=cols_to_drop)
    n_after = df.shape[1]

    report.append(f"Colonnes supprimées (>{MISSING_THRESHOLD*100:.0f}% NaN) : {len(cols_to_drop)}")
    report.append(f"Colonnes restantes : {n_after} (V1 en gardait {434-208}={434-208})")
    print(f"    V1 supprimait 208 colonnes → V2 supprime {len(cols_to_drop)} colonnes")
    print(f"    V1 gardait 226 colonnes → V2 garde {n_after} colonnes")

    # ── 2.2 Séparer features et label ────────────────────────
    print(f"  2.2 Séparation features / label...")
    y = df["isFraud"].values
    X = df.drop(columns=["isFraud", "TransactionID"])

    fraud_count = y.sum()
    report.append(f"\nFraudes : {fraud_count:,} / {len(y):,} ({fraud_count/len(y)*100:.2f}%)")
    print(f"    Fraudes : {fraud_count:,} ({fraud_count/len(y)*100:.2f}%)")

    # ── 2.3 Feature engineering temporel ─────────────────────
    print(f"  2.3 Feature engineering temporel...")
    if "TransactionDT" in X.columns:
        X["hour"] = (X["TransactionDT"] / 3600 % 24).astype(int)
        X["day_of_week"] = (X["TransactionDT"] / 86400 % 7).astype(int)
        X["is_night"] = ((X["hour"] >= 22) | (X["hour"] <= 6)).astype(int)
        X["is_peak_fraud_hour"] = ((X["hour"] >= 5) & (X["hour"] <= 10)).astype(int)
        report.append(f"\nFeatures temporelles ajoutées : hour, day_of_week, is_night, is_peak_fraud_hour")
        print(f"    +4 features : hour, day_of_week, is_night, is_peak_fraud_hour")

    # ── 2.4 Log-transformation du montant ────────────────────
    print(f"  2.4 Log-transformation du montant...")
    if "TransactionAmt" in X.columns:
        X["TransactionAmt_log"] = np.log1p(X["TransactionAmt"])
        X["TransactionAmt_is_round"] = (X["TransactionAmt"] % 1 == 0).astype(int)
        report.append(f"Features montant ajoutées : TransactionAmt_log, TransactionAmt_is_round")
        print(f"    +2 features : TransactionAmt_log, is_round")

    # ── 2.5 Identifier les types de colonnes ─────────────────
    print(f"  2.5 Identification des colonnes...")
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    print(f"    {len(numeric_cols)} numériques, {len(categorical_cols)} catégorielles")

    # ── 2.6 Indicateurs de valeurs manquantes ────────────────
    print(f"  2.6 Création des indicateurs de valeurs manquantes...")
    n_indicators = 0
    cols_with_missing = X.columns[X.isnull().mean() > MISSING_INDICATOR_THRESHOLD].tolist()

    for col in cols_with_missing:
        indicator_name = f"{col}_missing"
        X[indicator_name] = X[col].isnull().astype(int)
        n_indicators += 1

    report.append(f"\nIndicateurs de NaN créés : {n_indicators}")
    report.append(f"  (V1 n'en créait aucun — on perdait l'info 'valeur absente')")
    print(f"    +{n_indicators} colonnes indicatrices (V1 : 0)")

    # ── 2.7 Remplissage des NaN ──────────────────────────────
    print(f"  2.7 Remplissage des NaN...")
    for col in numeric_cols:
        if X[col].isnull().any():
            X[col] = X[col].fillna(X[col].median())

    for col in categorical_cols:
        X[col] = X[col].fillna("MISSING")

    # ── 2.8 Frequency encoding (remplace LabelEncoder) ──────
    print(f"  2.8 Frequency encoding des catégorielles...")
    freq_maps = {}
    for col in categorical_cols:
        freq = X[col].value_counts(normalize=True)
        freq_maps[col] = freq.to_dict()
        X[col] = X[col].map(freq).astype(float)

    report.append(f"\nFrequency encoding : {len(categorical_cols)} colonnes")
    report.append(f"  (V1 utilisait LabelEncoder — ordres arbitraires)")
    report.append(f"  (V2 utilise la fréquence — plus informatif)")
    print(f"    {len(categorical_cols)} colonnes encodées par fréquence")

    # Reidentifier toutes les colonnes numériques (après encoding)
    all_numeric = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()

    # ── 2.9 Normalisation ────────────────────────────────────
    print(f"  2.9 Normalisation StandardScaler...")
    scaler = StandardScaler()
    X[all_numeric] = scaler.fit_transform(X[all_numeric].values)

    remaining_nan = X.isnull().sum().sum()
    if remaining_nan > 0:
        print(f"    ⚠ {remaining_nan} NaN restants, remplissage par 0")
        X = X.fillna(0)

    print(f"    {len(all_numeric)} colonnes normalisées")
    print(f"    Total features : {X.shape[1]}")

    # ── 2.10 Split ───────────────────────────────────────────
    print(f"  2.10 Split train/val (80/20 stratifié)...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )

    report.append(f"\nSplit : Train {len(X_train):,} | Val {len(X_val):,}")
    report.append(f"Total features : {X.shape[1]}")
    print(f"    Train : {len(X_train):,} | Val : {len(X_val):,}")

    elapsed = time.time() - t0
    print(f"  ✓ Transform V2 en {elapsed:.1f}s\n")

    return {
        "X_train": X_train, "X_val": X_val,
        "y_train": y_train, "y_val": y_val,
        "feature_names": list(X.columns),
        "report": report,
    }


def load_v2(result, processed_dir=PROCESSED_DIR):
    """Phase 3 — Load : sauvegarde en parquet."""
    print("=" * 60)
    print("PHASE 3 — LOAD")
    print("=" * 60)

    # Sauvegarder dans des fichiers séparés (V2)
    train_df = result["X_train"].copy()
    train_df["isFraud"] = result["y_train"]
    train_path = os.path.join(processed_dir, "train_clean_v2.parquet")
    train_df.to_parquet(train_path, index=False)
    print(f"  → Train : {train_path} ({os.path.getsize(train_path)/1e6:.1f} Mo)")

    val_df = result["X_val"].copy()
    val_df["isFraud"] = result["y_val"]
    val_path = os.path.join(processed_dir, "val_clean_v2.parquet")
    val_df.to_parquet(val_path, index=False)
    print(f"  → Val   : {val_path} ({os.path.getsize(val_path)/1e6:.1f} Mo)")

    features_path = os.path.join(processed_dir, "feature_names_v2.txt")
    with open(features_path, "w") as f:
        f.write("\n".join(result["feature_names"]))

    report_path = os.path.join(processed_dir, "preprocessing_report_v2.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("PREPROCESSING V2 REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write("CHANGEMENTS PAR RAPPORT À V1 :\n")
        f.write("  - Seuil NaN : 70% -> 95%\n")
        f.write("  - Indicateurs de valeurs manquantes ajoutés\n")
        f.write("  - Frequency encoding au lieu de LabelEncoder\n")
        f.write("  - Features temporelles (hour, day, night, peak)\n")
        f.write("  - Log-transformation du montant\n\n")
        for line in result["report"]:
            f.write(line + "\n")

    print(f"  → Rapport : {report_path}")
    print(f"  ✓ Load terminé\n")


def run():
    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║  PREPROCESSING V2 — VERSION AMÉLIORÉE                    ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    total_t0 = time.time()

    df = extract(RAW_DIR)
    result = transform_v2(df)
    load_v2(result)

    print("=" * 60)
    print(f"PIPELINE V2 TERMINÉ en {time.time()-total_t0:.1f}s")
    print(f"  Données dans : {PROCESSED_DIR}/")
    print("=" * 60)

    # Comparaison V1 vs V2
    print(f"\n  COMPARAISON V1 → V2 :")
    print(f"  {'':15s} {'V1':>10s} {'V2':>10s}")
    print(f"  {'Seuil NaN':15s} {'70%':>10s} {'95%':>10s}")
    print(f"  {'Colonnes gardées':15s} {'226':>10s} {result['X_train'].shape[1]:>10d}")
    print(f"  {'Missing indicators':15s} {'0':>10s} {'oui':>10s}")
    print(f"  {'Encoding':15s} {'Label':>10s} {'Frequency':>10s}")
    print(f"  {'Features temps':15s} {'non':>10s} {'oui':>10s}")
    print(f"  {'Log montant':15s} {'non':>10s} {'oui':>10s}")


if __name__ == "__main__":
    run()
