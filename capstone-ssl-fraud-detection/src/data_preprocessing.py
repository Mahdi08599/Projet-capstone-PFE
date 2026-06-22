"""
=================================================================
Data Preprocessing Pipeline — Capstone SSL Fraud Detection
=================================================================
ETL léger pour le projet :
    Extract  → charger les CSV bruts, joindre transaction + identity
    Transform → nettoyage, encodage, normalisation, sélection de features
    Load     → sauvegarder les données propres en parquet

Usage :
    python src/data_preprocessing.py

Entrée  : data/raw/train_transaction.csv, data/raw/train_identity.csv
Sortie  : data/processed/train_clean.parquet
          data/processed/preprocessing_report.txt
=================================================================
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ─── Configuration ──────────────────────────────────────────────
RAW_DIR = os.path.join("data", "raw")
PROCESSED_DIR = os.path.join("data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Seuil : on supprime les colonnes avec plus de X% de valeurs manquantes
MISSING_THRESHOLD = 0.70

# Ratio de masquage pour le Self-Supervised Learning (utilisé plus tard)
MASK_RATIO = 0.15

# Random seed pour reproductibilité
SEED = 42


# =================================================================
# PHASE 1 — EXTRACT : Chargement et jointure
# =================================================================
def extract(raw_dir: str) -> pd.DataFrame:
    """
    Charge les fichiers CSV bruts et les fusionne sur TransactionID.
    
    Returns:
        DataFrame fusionné (transaction LEFT JOIN identity)
    """
    print("=" * 60)
    print("PHASE 1 — EXTRACT")
    print("=" * 60)
    
    t0 = time.time()
    
    # Charger les fichiers
    train_tx_path = os.path.join(raw_dir, "train_transaction.csv")
    train_id_path = os.path.join(raw_dir, "train_identity.csv")
    
    print(f"  Chargement de train_transaction.csv ...")
    train_tx = pd.read_csv(train_tx_path)
    print(f"    → {train_tx.shape[0]:,} lignes × {train_tx.shape[1]} colonnes")
    
    print(f"  Chargement de train_identity.csv ...")
    train_id = pd.read_csv(train_id_path)
    print(f"    → {train_id.shape[0]:,} lignes × {train_id.shape[1]} colonnes")
    
    # Fusion (LEFT JOIN sur TransactionID)
    print(f"  Fusion sur TransactionID (LEFT JOIN) ...")
    df = train_tx.merge(train_id, on="TransactionID", how="left")
    print(f"    → Résultat : {df.shape[0]:,} lignes × {df.shape[1]} colonnes")
    
    # Stats rapides
    n_with_identity = train_id["TransactionID"].nunique()
    pct = n_with_identity / train_tx.shape[0] * 100
    print(f"    → {pct:.1f}% des transactions ont des données d'identité")
    
    elapsed = time.time() - t0
    print(f"  ✓ Extract terminé en {elapsed:.1f}s\n")
    
    return df


# =================================================================
# PHASE 2 — TRANSFORM : Nettoyage et préparation
# =================================================================
def transform(df: pd.DataFrame, missing_threshold: float = MISSING_THRESHOLD) -> dict:
    """
    Pipeline de transformation complet :
    1. Suppression des colonnes trop vides
    2. Séparation features / label
    3. Identification des types de colonnes
    4. Traitement des valeurs manquantes
    5. Encodage des variables catégorielles
    6. Normalisation des variables numériques
    
    Returns:
        dict avec X_train, X_val, y_train, y_val, feature_names,
        numeric_cols, categorical_cols, scaler, label_encoders
    """
    print("=" * 60)
    print("PHASE 2 — TRANSFORM")
    print("=" * 60)
    
    t0 = time.time()
    report_lines = []
    
    # ── 2.1 Supprimer les colonnes trop vides ───────────────────
    print(f"  2.1 Suppression des colonnes avec >{missing_threshold*100:.0f}% de NaN ...")
    missing_pct = df.isnull().mean()
    cols_to_drop = missing_pct[missing_pct > missing_threshold].index.tolist()
    
    # On garde toujours TransactionID et isFraud
    cols_to_drop = [c for c in cols_to_drop if c not in ["TransactionID", "isFraud"]]
    
    df = df.drop(columns=cols_to_drop)
    report_lines.append(f"Colonnes supprimées (>{missing_threshold*100:.0f}% NaN) : {len(cols_to_drop)}")
    report_lines.append(f"  Exemples : {cols_to_drop[:10]}")
    print(f"    → {len(cols_to_drop)} colonnes supprimées")
    print(f"    → Restent {df.shape[1]} colonnes")
    
    # ── 2.2 Séparer features et label ───────────────────────────
    print(f"  2.2 Séparation features / label ...")
    y = df["isFraud"].values
    X = df.drop(columns=["isFraud", "TransactionID"])
    
    fraud_count = y.sum()
    fraud_pct = fraud_count / len(y) * 100
    report_lines.append(f"\nTarget isFraud :")
    report_lines.append(f"  Fraudes : {fraud_count:,} ({fraud_pct:.2f}%)")
    report_lines.append(f"  Légitimes : {len(y) - fraud_count:,} ({100-fraud_pct:.2f}%)")
    print(f"    → Fraudes : {fraud_count:,} ({fraud_pct:.2f}%) | Légitimes : {len(y)-fraud_count:,}")
    
    # ── 2.3 Identifier les types de colonnes ────────────────────
    print(f"  2.3 Identification des types de colonnes ...")
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    
    report_lines.append(f"\nTypes de colonnes :")
    report_lines.append(f"  Numériques : {len(numeric_cols)}")
    report_lines.append(f"  Catégorielles : {len(categorical_cols)}")
    print(f"    → {len(numeric_cols)} numériques, {len(categorical_cols)} catégorielles")
    
    # ── 2.4 Traiter les valeurs manquantes ──────────────────────
    print(f"  2.4 Traitement des valeurs manquantes ...")
    
    # Numériques : remplir par la médiane (robuste aux outliers)
    for col in numeric_cols:
        median_val = X[col].median()
        X[col] = X[col].fillna(median_val)
    
    # Catégorielles : remplir par "UNKNOWN"
    for col in categorical_cols:
        X[col] = X[col].fillna("UNKNOWN")
    
    remaining_nan = X.isnull().sum().sum()
    report_lines.append(f"\nValeurs manquantes restantes : {remaining_nan}")
    print(f"    → NaN restants après traitement : {remaining_nan}")
    
    # ── 2.5 Encoder les variables catégorielles ─────────────────
    print(f"  2.5 Encodage des variables catégorielles (LabelEncoder) ...")
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
    
    report_lines.append(f"\nEncodage catégoriel :")
    for col in categorical_cols:
        n_classes = len(label_encoders[col].classes_)
        report_lines.append(f"  {col} : {n_classes} classes")
    print(f"    → {len(categorical_cols)} colonnes encodées")
    
    # ── 2.6 Normaliser les variables numériques ─────────────────
    print(f"  2.6 Normalisation (StandardScaler) ...")
    all_feature_cols = numeric_cols + categorical_cols
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    print(f"    → {len(numeric_cols)} colonnes normalisées")
    
    # ── 2.7 Split train / validation ────────────────────────────
    print(f"  2.7 Split train/validation (80/20, stratifié) ...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.20,
        random_state=SEED,
        stratify=y  # important pour garder le ratio de fraude
    )
    
    report_lines.append(f"\nSplit train/validation :")
    report_lines.append(f"  Train : {len(X_train):,} lignes ({y_train.sum():,} fraudes)")
    report_lines.append(f"  Val   : {len(X_val):,} lignes ({y_val.sum():,} fraudes)")
    print(f"    → Train : {len(X_train):,} | Val : {len(X_val):,}")
    
    elapsed = time.time() - t0
    print(f"  ✓ Transform terminé en {elapsed:.1f}s\n")
    
    return {
        "X_train": X_train,
        "X_val": X_val,
        "y_train": y_train,
        "y_val": y_val,
        "feature_names": list(X.columns),
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "scaler": scaler,
        "label_encoders": label_encoders,
        "report": report_lines,
    }


# =================================================================
# PHASE 3 — LOAD : Sauvegarde des données propres
# =================================================================
def load(result: dict, processed_dir: str = PROCESSED_DIR):
    """
    Sauvegarde les données nettoyées en format parquet (rapide et compact).
    Génère aussi un rapport de preprocessing.
    """
    print("=" * 60)
    print("PHASE 3 — LOAD")
    print("=" * 60)
    
    t0 = time.time()
    
    # Sauvegarder X_train + y_train
    train_df = result["X_train"].copy()
    train_df["isFraud"] = result["y_train"]
    train_path = os.path.join(processed_dir, "train_clean.parquet")
    train_df.to_parquet(train_path, index=False)
    print(f"  → Train sauvé : {train_path} ({os.path.getsize(train_path)/1e6:.1f} Mo)")
    
    # Sauvegarder X_val + y_val
    val_df = result["X_val"].copy()
    val_df["isFraud"] = result["y_val"]
    val_path = os.path.join(processed_dir, "val_clean.parquet")
    val_df.to_parquet(val_path, index=False)
    print(f"  → Val sauvé   : {val_path} ({os.path.getsize(val_path)/1e6:.1f} Mo)")
    
    # Sauvegarder les noms de features (utile pour le modèle)
    features_path = os.path.join(processed_dir, "feature_names.txt")
    with open(features_path, "w") as f:
        f.write("\n".join(result["feature_names"]))
    print(f"  → Features    : {features_path}")
    
    # Sauvegarder le rapport
    report_path = os.path.join(processed_dir, "preprocessing_report.txt")
    with open(report_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("PREPROCESSING REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Date : {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Seed : {SEED}\n\n")
        for line in result["report"]:
            f.write(line + "\n")
    print(f"  → Rapport     : {report_path}")
    
    elapsed = time.time() - t0
    print(f"  ✓ Load terminé en {elapsed:.1f}s\n")


# =================================================================
# MAIN — Exécution du pipeline complet
# =================================================================
def run_pipeline():
    """Lance le pipeline ETL complet."""
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║  CAPSTONE SSL FRAUD DETECTION — DATA PIPELINE            ║")
    print("╚" + "═" * 58 + "╝\n")
    
    total_t0 = time.time()
    
    # Phase 1 : Extract
    df = extract(RAW_DIR)
    
    # Phase 2 : Transform
    result = transform(df)
    
    # Phase 3 : Load
    load(result)
    
    total_elapsed = time.time() - total_t0
    print("=" * 60)
    print(f"PIPELINE TERMINÉ en {total_elapsed:.1f}s")
    print(f"  Données prêtes dans : {PROCESSED_DIR}/")
    print("=" * 60)
    
    return result


if __name__ == "__main__":
    run_pipeline()
