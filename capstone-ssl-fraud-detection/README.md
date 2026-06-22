# Fraud Detection in Digital Banking using Self-Supervised Learning

**Capstone Project — Master Data / AI**

**Authors:** EL HAMDAOUI Mohamed, BEN ARAFI Mahdi  
**Supervisor:** Mahdi ZARG AYOUNA

---

## Project overview

This project explores whether **self-supervised learning** can improve fraud detection in digital banking when labeled transaction data is limited. The approach uses a two-step framework:

1. **Self-supervised pretraining** — Train an encoder to reconstruct randomly masked transaction features (no fraud labels needed)
2. **Supervised fine-tuning** — Reuse the pretrained encoder and add a classification head to predict fraud

## Dataset

**IEEE-CIS Fraud Detection Dataset** (Kaggle)

| File | Rows | Columns | Description |
|---|---|---|---|
| `train_transaction.csv` | 590,540 | 394 | Transaction features + `isFraud` label |
| `train_identity.csv` | 144,233 | 41 | Device and identity features |
| `test_transaction.csv` | 506,691 | 393 | Test transactions (no label) |
| `test_identity.csv` | 141,907 | 41 | Test identity features |

**Fraud rate:** 3.50% (highly imbalanced)

## Setup

### 1. Clone and enter the project

```bash
git clone <your-repo-url>
cd capstone-ssl-fraud-detection
```

### 2. Create a Python environment

With conda:
```bash
conda create -n ssl-fraud python=3.11
conda activate ssl-fraud
```

Or with venv:
```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

For PyTorch with GPU (CUDA), replace the torch line:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### 4. Download the dataset

Download from [Kaggle IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection/data) and place the CSV files in `data/raw/`:

```
data/raw/
├── train_transaction.csv
├── train_identity.csv
├── test_transaction.csv
├── test_identity.csv
└── sample_submission.csv
```

### 5. Run the preprocessing pipeline

```bash
python src/data_preprocessing.py
```

This produces cleaned data in `data/processed/`.

## Project structure

```
capstone-ssl-fraud-detection/
│
├── data/
│   ├── raw/                          # CSV bruts (non versionnés)
│   └── processed/                    # Données nettoyées (.parquet)
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_self_supervised_pretraining.ipynb
│   ├── 04_fraud_detection_finetuning.ipynb
│   └── 05_results_analysis.ipynb
│
├── src/
│   ├── data_preprocessing.py         # Pipeline ETL complet
│   ├── dataset.py                    # PyTorch datasets + masking
│   ├── ssl_model.py                  # Architecture de l'encodeur SSL
│   ├── train_ssl.py                  # Boucle d'entraînement SSL
│   ├── finetune.py                   # Fine-tuning classification fraude
│   └── evaluate.py                   # Métriques et visualisations
│
├── models/                           # Poids sauvegardés
├── reports/
│   ├── figures/                      # Graphiques pour le mémoire
│   └── results/                      # Tableaux de résultats
│
├── requirements.txt
├── .gitignore
└── README.md
```

## Pipeline

```
Dataset (IEEE-CIS)
  → Preprocessing (merge, clean, encode, normalize)
  → Feature masking (15% random)
  → Self-supervised pretraining (reconstruct masked features)
  → Fraud detection fine-tuning (encoder + classification head)
  → Evaluation (AUC-ROC, Precision, Recall, F1, Confusion Matrix)
```

## Evaluation metrics

- **AUC-ROC** — overall discrimination ability
- **Precision** — among predicted frauds, how many are real
- **Recall** — among real frauds, how many are detected (most important)
- **F1-score** — harmonic mean of precision and recall
- **Confusion matrix** — detailed error analysis
