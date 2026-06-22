"""
=================================================================
Pipeline complet : SSL + XGBoost + SSL-Guided Risk Scoring
=================================================================

Ce script se lance directement, comme pipeline_v2.py.

Il fait tout automatiquement :
  1. Charge train_clean_v2.parquet et val_clean_v2.parquet
  2. Entraîne un modèle SSL par masquage/reconstruction
  3. Extrait les embeddings SSL
  4. Calcule l'erreur de reconstruction SSL comme score d'anomalie
  5. Entraîne XGBoost sur les features originales + embeddings SSL + score SSL
  6. Combine :
        - probabilité XGBoost
        - score d'anomalie SSL
        - heuristique métier simple
     dans un score final de risque
  7. Optimise alpha/beta/gamma + seuil pour maximiser le F1-score
  8. Sauvegarde les résultats dans reports/results

Usage depuis la racine :
    python src/pipeline_ssl_guided_scoring.py

Usage depuis src :
    python .\pipeline_ssl_guided_scoring.py

Sorties :
    reports/results/ssl_guided_risk_results.txt
    reports/results/ssl_guided_risk_predictions.csv
    reports/results/ssl_guided_risk_summary.json
    reports/figures/ssl_guided_score_distribution.png
    models/ssl_guided_encoder.pt
=================================================================
"""

import os
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    recall_score,
    precision_score,
    confusion_matrix,
    classification_report,
)

warnings.filterwarnings("ignore")


# =================================================================
# CONFIG
# =================================================================

SEED = 42
SSL_EPOCHS = 20
SSL_BATCH = 512
SSL_LR = 1e-3
MASK_RATIO = 0.15

HIDDEN_DIM = 256
LATENT_DIM = 96

WEIGHT_STEP = 0.05
THRESHOLD_STEP = 0.01

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

np.random.seed(SEED)
torch.manual_seed(SEED)


def get_project_root() -> Path:
    """
    Détecte automatiquement la racine du projet,
    que le script soit lancé depuis src ou depuis la racine.
    """
    script_dir = Path(__file__).resolve().parent

    if script_dir.name.lower() == "src":
        return script_dir.parent

    return Path.cwd()


PROJECT_ROOT = get_project_root()

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
RESULTS_DIR = PROJECT_ROOT / "reports" / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# =================================================================
# MODELE SSL
# =================================================================

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, latent_dim=96):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(hidden_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim=96, hidden_dim=256, output_dim=743):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z):
        return self.net(z)


class SSLModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, latent_dim=96):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)

    def forward(self, x):
        z = self.encoder(x)
        x_reconstructed = self.decoder(z)
        return x_reconstructed


# =================================================================
# FONCTIONS UTILITAIRES
# =================================================================

def masked_mse_loss(pred, target, mask):
    """
    Loss SSL calculée seulement sur les colonnes masquées.
    """
    n = mask.sum()

    if n == 0:
        return torch.tensor(0.0, device=pred.device)

    return ((pred * mask - target * mask) ** 2).sum() / n


def train_ssl_model(X_train):
    """
    Entraîne le modèle SSL par masquage aléatoire de features.
    """
    n_features = X_train.shape[1]

    print("\n2. Entraînement SSL...")
    print(f"   Features : {n_features}")
    print(f"   Epochs   : {SSL_EPOCHS}")
    print(f"   Device   : {DEVICE}")

    model = SSLModel(
        input_dim=n_features,
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=SSL_LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=3,
        factor=0.5,
    )

    n_mask = max(1, int(n_features * MASK_RATIO))

    X_tensor = torch.FloatTensor(X_train)
    dataset = torch.utils.data.TensorDataset(X_tensor)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=SSL_BATCH,
        shuffle=True,
        num_workers=0,
    )

    train_losses = []

    for epoch in range(1, SSL_EPOCHS + 1):
        model.train()

        epoch_loss = 0.0
        n_batches = 0

        for (batch,) in loader:
            batch = batch.to(DEVICE)

            mask = torch.zeros_like(batch)

            for i in range(len(batch)):
                idx = torch.randperm(n_features, device=DEVICE)[:n_mask]
                mask[i, idx] = 1.0

            x_masked = batch.clone()
            x_masked[mask.bool()] = 0.0

            pred = model(x_masked)
            loss = masked_mse_loss(pred, batch, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        train_losses.append(avg_loss)

        scheduler.step(avg_loss)
        lr = optimizer.param_groups[0]["lr"]

        print(f"   Epoch {epoch:02d}/{SSL_EPOCHS} | Loss: {avg_loss:.6f} | LR: {lr:.6f}")

    torch.save(model.encoder.state_dict(), MODEL_DIR / "ssl_guided_encoder.pt")
    np.savez(MODEL_DIR / "ssl_guided_history.npz", train_losses=np.array(train_losses))

    print("   Encodeur SSL sauvegardé.")

    return model, train_losses


def extract_embeddings(encoder, X, batch_size=1024):
    """
    Extrait les embeddings SSL depuis l'encodeur.
    """
    encoder.eval()
    embeddings = []

    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.FloatTensor(X[i:i + batch_size]).to(DEVICE)
            z = encoder(batch).cpu().numpy()
            embeddings.append(z)

    return np.vstack(embeddings)


def compute_reconstruction_error(model, X, batch_size=1024):
    """
    Calcule l'erreur de reconstruction SSL pour chaque transaction.
    Plus l'erreur est forte, plus la transaction est atypique.
    """
    model.eval()
    errors = []

    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.FloatTensor(X[i:i + batch_size]).to(DEVICE)
            reconstructed = model(batch)

            batch_error = torch.mean((batch - reconstructed) ** 2, dim=1)
            errors.append(batch_error.cpu().numpy())

    return np.concatenate(errors)


def minmax_normalize(values):
    """
    Normalisation min-max entre 0 et 1.
    """
    values = np.asarray(values, dtype=float)

    min_val = np.nanmin(values)
    max_val = np.nanmax(values)

    if np.isclose(max_val, min_val):
        return np.zeros_like(values, dtype=float)

    return (values - min_val) / (max_val - min_val)


def get_classifier():
    """
    Utilise XGBoost si installé.
    Sinon, fallback vers GradientBoostingClassifier pour éviter que le script plante.
    """
    try:
        from xgboost import XGBClassifier

        clf = XGBClassifier(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0,
            reg_lambda=1,
            random_state=SEED,
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
        )

        return clf, "XGBoost"

    except Exception:
        clf = GradientBoostingClassifier(
            n_estimators=250,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_leaf=30,
            random_state=SEED,
        )

        return clf, "GradientBoosting fallback"


def find_best_threshold(y_true, scores):
    """
    Trouve le seuil qui maximise le F1-score.
    """
    best = {
        "threshold": 0.5,
        "f1": -1,
        "precision": 0,
        "recall": 0,
    }

    for threshold in np.arange(0.01, 0.99 + THRESHOLD_STEP, THRESHOLD_STEP):
        y_pred = (scores >= threshold).astype(int)

        f1 = f1_score(y_true, y_pred, zero_division=0)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)

        if f1 > best["f1"]:
            best = {
                "threshold": float(threshold),
                "f1": float(f1),
                "precision": float(precision),
                "recall": float(recall),
            }

    return best


def evaluate_prediction(y_true, scores, threshold=None):
    """
    Évalue une prédiction probabiliste.
    """
    if threshold is None:
        best_t = find_best_threshold(y_true, scores)
        threshold = best_t["threshold"]

    y_pred = (scores >= threshold).astype(int)

    metrics = {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    try:
        metrics["auc_roc"] = float(roc_auc_score(y_true, scores))
    except Exception:
        metrics["auc_roc"] = None

    try:
        metrics["auc_pr"] = float(average_precision_score(y_true, scores))
    except Exception:
        metrics["auc_pr"] = None

    return metrics


def build_heuristic_score(val_df):
    """
    Heuristique métier simple, sans fichier externe.

    Règles utilisées :
    - montant très élevé si TransactionAmt existe ;
    - score SSL déjà pris séparément, donc ici l'heuristique reste volontairement légère.

    Si TransactionAmt n'existe pas dans les données nettoyées, score = 0.
    """
    heuristic = np.zeros(len(val_df), dtype=float)
    used_rules = 0

    if "TransactionAmt" in val_df.columns:
        amount = val_df["TransactionAmt"].values.astype(float)
        threshold = np.nanpercentile(amount, 95)
        heuristic += (amount > threshold).astype(float)
        used_rules += 1

    if used_rules == 0:
        return heuristic

    return heuristic / used_rules


def optimize_guided_score(y_true, xgb_proba, ssl_anomaly_score, heuristic_score):
    """
    Optimise :
      FinalScore = alpha * XGBoostProb
                 + beta  * SSLAnomalyScore
                 + gamma * HeuristicScore

    avec alpha + beta + gamma = 1.
    """
    best = {
        "f1": -1,
        "precision": None,
        "recall": None,
        "auc_roc": None,
        "auc_pr": None,
        "threshold": None,
        "alpha": None,
        "beta": None,
        "gamma": None,
        "confusion_matrix": None,
    }

    weights = np.arange(0, 1 + WEIGHT_STEP, WEIGHT_STEP)

    heuristic_available = not np.allclose(heuristic_score, 0)

    for alpha in weights:
        for beta in weights:
            gamma = 1 - alpha - beta

            if gamma < -1e-9:
                continue

            gamma = max(0, gamma)

            if not heuristic_available and gamma > 1e-9:
                continue

            if np.isclose(alpha + beta + gamma, 0):
                continue

            final_score = (
                alpha * xgb_proba
                + beta * ssl_anomaly_score
                + gamma * heuristic_score
            )

            metrics = evaluate_prediction(y_true, final_score)

            if metrics["f1"] > best["f1"]:
                best.update({
                    "f1": metrics["f1"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "auc_roc": metrics["auc_roc"],
                    "auc_pr": metrics["auc_pr"],
                    "threshold": metrics["threshold"],
                    "alpha": float(alpha),
                    "beta": float(beta),
                    "gamma": float(gamma),
                    "confusion_matrix": metrics["confusion_matrix"],
                })

    final_score = (
        best["alpha"] * xgb_proba
        + best["beta"] * ssl_anomaly_score
        + best["gamma"] * heuristic_score
    )

    return best, final_score


def save_score_distribution(y_true, final_score):
    """
    Sauvegarde un graphique de distribution des scores finaux.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(final_score[y_true == 0], bins=50, alpha=0.6, label="Non fraude")
    ax.hist(final_score[y_true == 1], bins=50, alpha=0.6, label="Fraude")

    ax.set_title("Distribution du score final SSL-Guided")
    ax.set_xlabel("Final fraud risk score")
    ax.set_ylabel("Nombre de transactions")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "ssl_guided_score_distribution.png", dpi=150)
    plt.close()


def write_results_txt(summary, output_path):
    """
    Sauvegarde un fichier texte propre pour le mémoire.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("SSL-GUIDED FRAUD RISK SCORING RESULTS\n")
        f.write("=" * 60 + "\n\n")

        f.write("1. APPROCHE\n")
        f.write("- SSL apprend des représentations transactionnelles.\n")
        f.write("- L'erreur de reconstruction SSL est utilisée comme score d'anomalie.\n")
        f.write("- XGBoost prédit une probabilité de fraude.\n")
        f.write("- Un score final combine XGBoost + SSL anomaly + heuristique.\n")
        f.write("- Les poids et le seuil sont optimisés pour maximiser le F1-score.\n\n")

        f.write("2. RESULTATS\n")
        for name, metrics in summary["metrics"].items():
            f.write(f"\n{name}\n")
            f.write("-" * len(name) + "\n")
            f.write(f"AUC-ROC   : {metrics['auc_roc']:.4f}\n" if metrics["auc_roc"] is not None else "AUC-ROC   : None\n")
            f.write(f"AUC-PR    : {metrics['auc_pr']:.4f}\n" if metrics["auc_pr"] is not None else "AUC-PR    : None\n")
            f.write(f"Precision : {metrics['precision']:.4f}\n")
            f.write(f"Recall    : {metrics['recall']:.4f}\n")
            f.write(f"F1-score  : {metrics['f1']:.4f}\n")
            f.write(f"Threshold : {metrics['threshold']:.4f}\n")
            f.write(f"Confusion matrix : {metrics['confusion_matrix']}\n")

        f.write("\n3. MEILLEURE FUSION SSL-GUIDED\n")
        f.write(f"Alpha XGBoost    : {summary['best_guided']['alpha']:.2f}\n")
        f.write(f"Beta SSL anomaly : {summary['best_guided']['beta']:.2f}\n")
        f.write(f"Gamma heuristic  : {summary['best_guided']['gamma']:.2f}\n")
        f.write(f"Best threshold   : {summary['best_guided']['threshold']:.2f}\n")

        f.write("\n4. INTERPRETATION COURTE\n")
        f.write(
            "Le score SSL-Guided exploite le SSL au-delà des embeddings : "
            "l'erreur de reconstruction devient un signal d'anomalie. "
            "Ce signal est combiné avec la probabilité XGBoost afin d'améliorer "
            "l'équilibre entre précision et rappel, mesuré par le F1-score.\n"
        )


# =================================================================
# MAIN
# =================================================================

def run():
    start = time.time()

    print("=" * 70)
    print("PIPELINE : SSL + XGBOOST + SSL-GUIDED RISK SCORING")
    print("=" * 70)

    print(f"\nProject root : {PROJECT_ROOT}")
    print(f"Data dir     : {DATA_DIR}")
    print(f"Device       : {DEVICE}")

    train_path = DATA_DIR / "train_clean_v2.parquet"
    val_path = DATA_DIR / "val_clean_v2.parquet"

    if not train_path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {train_path}")

    if not val_path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {val_path}")

    # ── 1. Chargement des données ─────────────────────────────
    print("\n1. Chargement des données...")
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)

    if "isFraud" not in train_df.columns or "isFraud" not in val_df.columns:
        raise ValueError("La colonne isFraud doit exister dans train_clean_v2.parquet et val_clean_v2.parquet.")

    y_train = train_df["isFraud"].values.astype(int)
    X_train = train_df.drop(columns=["isFraud"]).values.astype(np.float32)

    y_val = val_df["isFraud"].values.astype(int)
    X_val = val_df.drop(columns=["isFraud"]).values.astype(np.float32)

    n_features = X_train.shape[1]

    print(f"   Train : {len(X_train):,} lignes")
    print(f"   Val   : {len(X_val):,} lignes")
    print(f"   Features : {n_features}")
    print(f"   Taux fraude train : {y_train.mean():.4f}")
    print(f"   Taux fraude val   : {y_val.mean():.4f}")

    # ── 2. Entrainement SSL ───────────────────────────────────
    ssl_model, train_losses = train_ssl_model(X_train)

    # ── 3. Embeddings + reconstruction error ──────────────────
    print("\n3. Extraction des embeddings et erreurs SSL...")

    Z_train = extract_embeddings(ssl_model.encoder, X_train)
    Z_val = extract_embeddings(ssl_model.encoder, X_val)

    train_reconstruction_error = compute_reconstruction_error(ssl_model, X_train)
    val_reconstruction_error = compute_reconstruction_error(ssl_model, X_val)

    train_ssl_anomaly = minmax_normalize(train_reconstruction_error)
    val_ssl_anomaly = minmax_normalize(val_reconstruction_error)

    print(f"   Embeddings train : {Z_train.shape}")
    print(f"   Embeddings val   : {Z_val.shape}")

    # ── 4. Features finales ───────────────────────────────────
    print("\n4. Construction des features finales...")

    X_train_ssl = np.hstack([
        X_train,
        Z_train,
        train_ssl_anomaly.reshape(-1, 1),
    ])

    X_val_ssl = np.hstack([
        X_val,
        Z_val,
        val_ssl_anomaly.reshape(-1, 1),
    ])

    print(f"   Features originales : {X_train.shape[1]}")
    print(f"   Embeddings SSL      : {Z_train.shape[1]}")
    print(f"   Score anomalie SSL  : 1")
    print(f"   Total               : {X_train_ssl.shape[1]}")

    # ── 5. Baseline XGBoost sur features originales ───────────
    print("\n5. Entraînement baseline sur features originales...")

    baseline_clf, baseline_name = get_classifier()
    print(f"   Classifier : {baseline_name}")

    baseline_clf.fit(X_train, y_train)
    baseline_proba = baseline_clf.predict_proba(X_val)[:, 1]
    baseline_metrics = evaluate_prediction(y_val, baseline_proba)

    print(
        f"   Baseline | AUC={baseline_metrics['auc_roc']:.4f} "
        f"F1={baseline_metrics['f1']:.4f} "
        f"Recall={baseline_metrics['recall']:.4f} "
        f"Precision={baseline_metrics['precision']:.4f}"
    )

    # ── 6. XGBoost sur features originales + SSL ──────────────
    print("\n6. Entraînement hybride sur features originales + SSL...")

    hybrid_clf, hybrid_name = get_classifier()
    print(f"   Classifier : {hybrid_name}")

    hybrid_clf.fit(X_train_ssl, y_train)
    hybrid_proba = hybrid_clf.predict_proba(X_val_ssl)[:, 1]
    hybrid_metrics = evaluate_prediction(y_val, hybrid_proba)

    print(
        f"   Hybrid SSL | AUC={hybrid_metrics['auc_roc']:.4f} "
        f"F1={hybrid_metrics['f1']:.4f} "
        f"Recall={hybrid_metrics['recall']:.4f} "
        f"Precision={hybrid_metrics['precision']:.4f}"
    )

    # ── 7. SSL-guided risk scoring ────────────────────────────
    print("\n7. Optimisation SSL-Guided Risk Scoring...")

    xgb_proba_norm = minmax_normalize(hybrid_proba)
    ssl_anomaly_norm = minmax_normalize(val_ssl_anomaly)
    heuristic_score = build_heuristic_score(val_df.drop(columns=["isFraud"]))

    guided_metrics, final_score = optimize_guided_score(
        y_true=y_val,
        xgb_proba=xgb_proba_norm,
        ssl_anomaly_score=ssl_anomaly_norm,
        heuristic_score=heuristic_score,
    )

    print(
        f"   Guided | AUC={guided_metrics['auc_roc']:.4f} "
        f"F1={guided_metrics['f1']:.4f} "
        f"Recall={guided_metrics['recall']:.4f} "
        f"Precision={guided_metrics['precision']:.4f}"
    )

    print(
        f"   Best weights : alpha={guided_metrics['alpha']:.2f}, "
        f"beta={guided_metrics['beta']:.2f}, "
        f"gamma={guided_metrics['gamma']:.2f}, "
        f"threshold={guided_metrics['threshold']:.2f}"
    )

    final_pred = (final_score >= guided_metrics["threshold"]).astype(int)

    # ── 8. Sauvegarde des prédictions ─────────────────────────
    print("\n8. Sauvegarde des résultats...")

    predictions_df = pd.DataFrame({
        "y_true": y_val,
        "baseline_proba": baseline_proba,
        "hybrid_xgb_proba": hybrid_proba,
        "ssl_anomaly_score": ssl_anomaly_norm,
        "heuristic_score": heuristic_score,
        "final_fraud_score": final_score,
        "final_prediction": final_pred,
    })

    predictions_path = RESULTS_DIR / "ssl_guided_risk_predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)

    summary = {
        "classifier_baseline": baseline_name,
        "classifier_hybrid": hybrid_name,
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_original_features": int(X_train.shape[1]),
        "n_ssl_embeddings": int(Z_train.shape[1]),
        "ssl_epochs": SSL_EPOCHS,
        "mask_ratio": MASK_RATIO,
        "best_guided": guided_metrics,
        "metrics": {
            "Baseline original features": baseline_metrics,
            "Hybrid original + SSL": hybrid_metrics,
            "SSL-Guided Risk Scoring": guided_metrics,
        },
    }

    summary_path = RESULTS_DIR / "ssl_guided_risk_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    txt_path = RESULTS_DIR / "ssl_guided_risk_results.txt"
    write_results_txt(summary, txt_path)

    save_score_distribution(y_val, final_score)

    # ── 9. Affichage final ────────────────────────────────────
    print("\n" + "=" * 70)
    print("RESULTATS FINAUX")
    print("=" * 70)

    rows = [
        ("Baseline original features", baseline_metrics),
        ("Hybrid original + SSL", hybrid_metrics),
        ("SSL-Guided Risk Scoring", guided_metrics),
    ]

    print(f"{'Approche':<32} | {'AUC':>8} | {'AUC-PR':>8} | {'Precision':>9} | {'Recall':>8} | {'F1':>8}")
    print("-" * 90)

    for name, m in rows:
        auc = m["auc_roc"] if m["auc_roc"] is not None else 0
        auc_pr = m["auc_pr"] if m["auc_pr"] is not None else 0

        print(
            f"{name:<32} | "
            f"{auc:>8.4f} | "
            f"{auc_pr:>8.4f} | "
            f"{m['precision']:>9.4f} | "
            f"{m['recall']:>8.4f} | "
            f"{m['f1']:>8.4f}"
        )

    print("\nMatrice de confusion SSL-Guided :")
    print(np.array(guided_metrics["confusion_matrix"]))

    print("\nClassification report SSL-Guided :")
    print(classification_report(y_val, final_pred, zero_division=0))

    print("\nFichiers créés :")
    print(f"   {predictions_path}")
    print(f"   {summary_path}")
    print(f"   {txt_path}")
    print(f"   {FIGURES_DIR / 'ssl_guided_score_distribution.png'}")
    print(f"   {MODEL_DIR / 'ssl_guided_encoder.pt'}")

    print(f"\nTemps total : {(time.time() - start) / 60:.2f} minutes")
    print("=" * 70)
    print("PIPELINE TERMINE")
    print("=" * 70)


if __name__ == "__main__":
    run()
