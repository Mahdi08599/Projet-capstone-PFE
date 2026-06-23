"""
=================================================================
Dashboard Streamlit — Fraud Detection (Modèle Final Optimisé)
=================================================================
Synchronisé avec le modèle XGBoost optimisé :
  AUC=0.9718 | F1=0.8210 | Precision=0.8797 | Recall=0.7697
  Seuil optimal = 0.56

Usage :
    pip install streamlit plotly xgboost scikit-learn
    streamlit run app_dashboard.py
=================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import unicodedata
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

DATA_RAW = os.path.join("data", "raw")
DATA_PROC = os.path.join("data", "processed")
MODELS_DIR = "models"
RESULTS_DIR = os.path.join("reports", "results")
FIGURES_DIR = os.path.join("reports", "figures")

FINAL_METRICS = {
    "auc": 0.9718, "avg_precision": 0.8608, "f1": 0.8210,
    "precision": 0.8797, "recall": 0.7697, "threshold": 0.56,
}
BEST_PARAMS = {
    "colsample_bytree": 0.736, "gamma": 0.057, "learning_rate": 0.149,
    "max_depth": 9, "min_child_weight": 3, "n_estimators": 562,
    "reg_alpha": 0.660, "reg_lambda": 2.134, "subsample": 0.822,
}
PROBLEMATIQUE = (
    "Comment concevoir un système hybride de détection de fraude bancaire digitale, "
    "combinant self-supervised learning, apprentissage supervisé et optimisation métier, "
    "afin d'améliorer la performance, l'explicabilité et l'exploitabilité des décisions "
    "dans un contexte de données fortement déséquilibrées ?"
)
CHATBOT_SOURCES = {
    "KPIs métier": "kpi_report.txt",
    "Modèle final": "final_model_report.txt",
    "Seuil métier": "base_threshold_business_strategy.txt",
    "Hyperparamètres": "hyperparameter_results.txt",
    "Comparaison modèles": "comparison_results.txt",
    "Labels limités": "limited_labels_results.txt",
    "Risque SSL guidé": "ssl_guided_risk_results.txt",
}


def money_usd(value, decimals=0):
    return f"{value:,.{decimals}f} USD".replace(",", " ")

st.set_page_config(page_title="Fraud Detection SSL — Capstone", page_icon="🛡️",
                   layout="wide", initial_sidebar_state="expanded")


@st.cache_data
def load_raw_data():
    tx = pd.read_csv(os.path.join(DATA_RAW, "train_transaction.csv"))
    ident = pd.read_csv(os.path.join(DATA_RAW, "train_identity.csv"))
    return tx.merge(ident, on="TransactionID", how="left")


@st.cache_data
def load_predictions():
    path = os.path.join(RESULTS_DIR, "tuned_predictions.npz")
    if os.path.exists(path):
        data = np.load(path)
        return data["y_true"], data["y_proba"]
    return None, None


@st.cache_data
def load_transaction_lookup():
    tx_cols = [
        "TransactionID", "isFraud", "TransactionDT", "TransactionAmt", "ProductCD",
        "card4", "card6", "P_emaildomain", "R_emaildomain",
    ]
    tx = pd.read_csv(os.path.join(DATA_RAW, "train_transaction.csv"), usecols=tx_cols)
    identity_path = os.path.join(DATA_RAW, "train_identity.csv")
    if os.path.exists(identity_path):
        id_head = pd.read_csv(identity_path, nrows=0)
        id_cols = [col for col in ["TransactionID", "DeviceType", "DeviceInfo"] if col in id_head.columns]
        if "TransactionID" in id_cols:
            ident = pd.read_csv(identity_path, usecols=id_cols)
            tx = tx.merge(ident, on="TransactionID", how="left")
    tx["hour"] = (tx["TransactionDT"] / 3600 % 24).astype(int)
    return tx


def normalize_text(text):
    text = text.lower()
    text = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in text if unicodedata.category(ch) != "Mn")


@st.cache_data
def load_chatbot_sources():
    sources = {}
    for label, filename in CHATBOT_SOURCES.items():
        path = os.path.join(RESULTS_DIR, filename)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8", errors="replace") as file:
                sources[label] = file.read()
    return sources


def search_report_excerpts(question, sources, max_results=2):
    stopwords = {
        "dans", "avec", "pour", "quoi", "quel", "quelle", "comment", "nous",
        "vous", "plus", "moins", "cette", "cela", "faire", "peut", "etre",
        "sont", "notre", "votre", "modele", "fraude", "fraudes",
    }
    tokens = {
        token for token in normalize_text(question).replace("?", " ").split()
        if len(token) >= 4 and token not in stopwords
    }
    scored_blocks = []
    for source_name, text in sources.items():
        blocks = [block.strip() for block in text.split("\n\n") if len(block.strip()) > 80]
        for block in blocks:
            norm_block = normalize_text(block)
            score = sum(1 for token in tokens if token in norm_block)
            if score > 0:
                scored_blocks.append((score, source_name, block))
    scored_blocks.sort(key=lambda item: item[0], reverse=True)
    return scored_blocks[:max_results]


def answer_transaction_lookup(question):
    id_match = re.search(r"\b(?:transactionid|transaction|id)\D*(\d{6,})\b", normalize_text(question))
    if not id_match:
        id_match = re.search(r"\b(\d{7,})\b", question)
    if not id_match:
        return None

    transaction_id = int(id_match.group(1))
    df = load_transaction_lookup()
    row = df[df["TransactionID"] == transaction_id]
    if row.empty:
        return (
            f"Je n'ai pas trouvé la transaction {transaction_id} dans le dataset brut IEEE-CIS utilisé par le projet. "
            "Vérifie l'ID ou utilise un ID présent dans train_transaction.csv.",
            ["Dataset brut"],
        )

    row = row.iloc[0]
    status = "Fraude observée" if int(row["isFraud"]) == 1 else "Transaction légitime observée"
    product = row["ProductCD"]
    product_df = df[df["ProductCD"] == product]
    product_rate = product_df["isFraud"].mean() * 100
    product_frauds = int(product_df["isFraud"].sum())
    product_total = len(product_df)
    email = row.get("P_emaildomain")
    device = row.get("DeviceType")

    response = (
        f"Résultat pour la transaction {transaction_id} :\n\n"
        f"- Statut réel dans le dataset : **{status}**.\n"
        f"- Produit : **{product}**.\n"
        f"- Montant : **{money_usd(row['TransactionAmt'], 2)}**.\n"
        f"- Heure estimée : **{int(row['hour'])}h**.\n"
        f"- Type de carte : **{row.get('card6', 'non renseigné')}**.\n"
        f"- Réseau carte : **{row.get('card4', 'non renseigné')}**.\n"
        f"- Email émetteur : **{email if pd.notna(email) else 'non renseigné'}**.\n"
        f"- Device : **{device if pd.notna(device) else 'non renseigné'}**.\n\n"
        f"Lecture produit : le produit {product} représente {product_total:,} transactions, "
        f"avec {product_frauds:,} fraudes observées, soit un taux de fraude de {product_rate:.2f}%.\n\n"
        "Important : cette réponse décrit les données observées dans le dataset. Elle ne remplace pas "
        "un scoring individuel du modèle final."
    )
    return response, ["Dataset brut", "KPIs produit calculés"]


def answer_product_lookup(question):
    q = normalize_text(question)
    product_match = re.search(r"\b(?:produit|product|productcd)\s*([wchrs])\b", q)
    if not product_match:
        return None

    product = product_match.group(1).upper()
    df = load_transaction_lookup()
    prod = df[df["ProductCD"] == product]
    if prod.empty:
        return (
            f"Je n'ai pas trouvé le produit {product} dans le dataset. Les produits disponibles sont W, C, H, R et S.",
            ["Dataset brut"],
        )

    total = len(prod)
    frauds = int(prod["isFraud"].sum())
    fraud_rate = frauds / total * 100
    fraud_amount = prod.loc[prod["isFraud"] == 1, "TransactionAmt"].sum()
    avg_fraud_amount = prod.loc[prod["isFraud"] == 1, "TransactionAmt"].mean()
    share_transactions = total / len(df) * 100
    share_frauds = frauds / df["isFraud"].sum() * 100

    response = (
        f"Analyse du produit {product} :\n\n"
        f"- Transactions totales : **{total:,}** ({share_transactions:.1f}% du dataset).\n"
        f"- Fraudes observées : **{frauds:,}** ({share_frauds:.1f}% des fraudes).\n"
        f"- Taux de fraude : **{fraud_rate:.2f}%**.\n"
        f"- Montant frauduleux total : **{money_usd(fraud_amount)}**.\n"
        f"- Montant moyen d'une fraude : **{money_usd(avg_fraud_amount)}**.\n\n"
    )
    if product == "W":
        response += (
            "Lecture métier : W a un taux de fraude faible, mais un volume très élevé. "
            "Il peut donc générer beaucoup de fraudes en valeur absolue malgré un risque relatif faible."
        )
    elif product == "C":
        response += (
            "Lecture métier : C est le produit le plus risqué en taux de fraude. "
            "Il doit être surveillé en priorité dans une stratégie de scoring."
        )
    else:
        response += (
            "Lecture métier : ce produit doit être interprété en comparant son taux de fraude et son volume. "
            "Un risque faible en pourcentage peut rester important si le volume est élevé."
        )
    return response, ["Dataset brut", "KPIs produit calculés"]


def answer_sample_analysis(question):
    q = normalize_text(question)
    wants_analysis = any(word in q for word in ["analyse", "analyses", "analyser", "resume", "synthese"])
    mentions_transactions = any(word in q for word in ["transaction", "transactions", "dataset", "echantillon", "nombre"])
    if not (wants_analysis and mentions_transactions):
        return None

    df = load_transaction_lookup()
    sample_size = 10000
    sample = df.sample(n=min(sample_size, len(df)), random_state=42)
    frauds = int(sample["isFraud"].sum())
    fraud_rate = frauds / len(sample) * 100
    amount_total = sample["TransactionAmt"].sum()
    fraud_amount = sample.loc[sample["isFraud"] == 1, "TransactionAmt"].sum()
    avg_amount = sample["TransactionAmt"].mean()
    avg_fraud_amount = sample.loc[sample["isFraud"] == 1, "TransactionAmt"].mean()
    product_stats = (
        sample.groupby("ProductCD")
        .agg(transactions=("isFraud", "count"), fraudes=("isFraud", "sum"))
        .reset_index()
    )
    product_stats["taux"] = product_stats["fraudes"] / product_stats["transactions"] * 100
    top_rate = product_stats.sort_values("taux", ascending=False).iloc[0]
    top_volume = product_stats.sort_values("fraudes", ascending=False).iloc[0]
    hour_stats = (
        sample.groupby("hour")
        .agg(transactions=("isFraud", "count"), fraudes=("isFraud", "sum"))
        .reset_index()
    )
    hour_stats["taux"] = hour_stats["fraudes"] / hour_stats["transactions"] * 100
    top_hour = hour_stats.sort_values("taux", ascending=False).iloc[0]

    response = (
        "J'ai choisi automatiquement un échantillon reproductible de "
        f"**{len(sample):,} transactions** du dataset pour donner une lecture rapide.\n\n"
        f"- Fraudes observées : **{frauds:,}**, soit **{fraud_rate:.2f}%**.\n"
        f"- Volume financier total : **{money_usd(amount_total)}**.\n"
        f"- Volume frauduleux observé : **{money_usd(fraud_amount)}**.\n"
        f"- Montant moyen transaction : **{money_usd(avg_amount)}**.\n"
        f"- Montant moyen fraude : **{money_usd(avg_fraud_amount)}**.\n"
        f"- Produit le plus risqué en taux : **{top_rate['ProductCD']}** "
        f"({top_rate['taux']:.2f}% de fraude dans l'échantillon).\n"
        f"- Produit qui génère le plus de fraudes : **{top_volume['ProductCD']}** "
        f"({int(top_volume['fraudes']):,} fraudes observées).\n"
        f"- Heure la plus risquée dans l'échantillon : **{int(top_hour['hour'])}h** "
        f"({top_hour['taux']:.2f}% de fraude).\n\n"
        "Lecture métier : cette analyse montre pourquoi une banque digitale doit raisonner à la fois en taux de risque "
        "et en volume. Un segment peut avoir un taux faible mais générer beaucoup d'alertes s'il concentre énormément de transactions."
    )
    return response, ["Dataset brut", "Échantillon reproductible random_state=42"]


def compute_rule_based_risk(amount, product, card_type, card_net, hour, email="autre", device="inconnu"):
    risk = 0
    factors = []
    prod_risk = {"C": 11.7, "S": 5.9, "H": 4.8, "R": 3.8, "W": 2.0}
    if prod_risk.get(product, 3.5) > 5:
        risk += 25
        factors.append(f"Produit {product} à haut risque ({prod_risk[product]:.1f}%)")
    elif prod_risk.get(product, 3.5) > 3.5:
        risk += 10
        factors.append(f"Produit {product} au-dessus du risque moyen")

    if 5 <= hour <= 10:
        risk += 20
        factors.append(f"Heure {hour}h dans le pic de fraude")
    if hour >= 22 or hour <= 6:
        risk += 8
        factors.append("Transaction nocturne")

    email_risk = {"mail.com": 19.0, "outlook.com": 9.5, "hotmail.com": 5.3, "gmail.com": 4.4}
    er = email_risk.get(email, 3.0)
    if er > 8:
        risk += 25
        factors.append(f"Email {email} très risqué ({er:.1f}%)")
    elif er > 5:
        risk += 10
        factors.append(f"Email {email} risqué ({er:.1f}%)")

    if device == "mobile":
        risk += 15
        factors.append("Mobile (10.2% vs 6.5% desktop)")
    if card_type == "credit":
        risk += 10
        factors.append("Carte crédit (6.7% vs 2.4% débit)")
    if card_net == "discover":
        risk += 10
        factors.append("Réseau Discover (7.7%)")
    if amount > 500:
        risk += 8
        factors.append(f"Montant élevé ({money_usd(amount)})")

    risk = min(risk, 100)
    if risk >= 70:
        decision = "SUSPECTE — blocage ou revue immédiate recommandé"
    elif risk >= 40:
        decision = "SURVEILLANCE — vérification conseillée"
    else:
        decision = "NORMALE — autorisation possible"
    return risk, decision, factors


def answer_conversational_transaction(question):
    q = normalize_text(question)
    if "transaction" not in q:
        return None
    if re.search(r"\b(?:transactionid|id)\D*\d{6,}\b", q):
        return None

    amount_match = re.search(r"(\d+(?:[.,]\d+)?)\s*(?:\$|dollar|usd)", q)
    if not amount_match:
        amount_match = re.search(r"(?:montant|transaction de)\s*(\d+(?:[.,]\d+)?)", q)
    product_match = re.search(r"\b(?:produit|product)\s*([wchrs])\b", q)
    hour_match = re.search(r"\b(?:a|à|vers)?\s*(\d{1,2})\s*h\b", q)

    if not (amount_match or product_match or hour_match):
        return None

    amount = float(amount_match.group(1).replace(",", ".")) if amount_match else 120.0
    product = product_match.group(1).upper() if product_match else "W"
    hour = int(hour_match.group(1)) if hour_match else 14
    hour = max(0, min(hour, 23))

    card_type = "credit" if any(word in q for word in ["credit", "crédit"]) else "debit" if any(word in q for word in ["debit", "débit"]) else "debit"
    if "discover" in q:
        card_net = "discover"
    elif "mastercard" in q:
        card_net = "mastercard"
    elif "american" in q:
        card_net = "american express"
    else:
        card_net = "visa"

    email = "autre"
    for domain in ["mail.com", "outlook.com", "hotmail.com", "gmail.com", "icloud.com"]:
        if domain in q:
            email = domain
            break
    device = "mobile" if "mobile" in q else "desktop" if "desktop" in q else "inconnu"

    risk, decision, factors = compute_rule_based_risk(amount, product, card_type, card_net, hour, email, device)
    factors_text = "\n".join(f"- {factor}" for factor in factors) if factors else "- Aucun facteur majeur détecté"
    response = (
        "Analyse conversationnelle de la transaction :\n\n"
        f"- Montant interprété : **{money_usd(amount, 2)}**.\n"
        f"- Produit : **{product}**.\n"
        f"- Carte : **{card_type}** / réseau **{card_net}**.\n"
        f"- Heure : **{hour}h**.\n"
        f"- Email : **{email}**.\n"
        f"- Device : **{device}**.\n\n"
        f"Score de risque estimé : **{risk}/100**.\n"
        f"Décision suggérée : **{decision}**.\n\n"
        "Facteurs détectés :\n"
        f"{factors_text}\n\n"
        "Important : ce score reprend la logique explicative de la page « Test du modèle ». "
        "Il sert à la démonstration métier et ne remplace pas le scoring complet XGBoost sur les 743 variables."
    )
    return response, ["Règles explicatives du dashboard", "KPIs métier"]


def answer_scenario_simulation(question):
    q = normalize_text(question)
    percent_match = re.search(r"(\d{1,3})\s*%", q)
    wants_scenario = any(word in q for word in ["si", "scenario", "simule", "detecte", "detecter", "recall"])
    if not (percent_match and wants_scenario):
        return None

    recall = min(max(int(percent_match.group(1)) / 100, 0.01), 0.99)
    cost_match = re.search(r"(?:cout|coût|investigation)\D*(\d{1,4})", question.lower())
    cost_inv = float(cost_match.group(1)) if cost_match else 15.0

    df = load_transaction_lookup()
    fraud = df[df["isFraud"] == 1]
    fraud_count = len(fraud)
    fraud_amount = fraud["TransactionAmt"].sum()
    detected = int(fraud_count * recall)
    saved = fraud_amount * recall
    false_alerts = int(detected * (1 - FINAL_METRICS["precision"]) / FINAL_METRICS["precision"])
    investigation_cost = (detected + false_alerts) * cost_inv
    net = saved - investigation_cost

    response = (
        f"Simulation simplifiée avec **{recall:.0%} de fraudes détectées** :\n\n"
        f"- Fraudes détectées estimées : **{detected:,} / {fraud_count:,}**.\n"
        f"- Montant frauduleux sauvé estimé : **{money_usd(saved)}**.\n"
        f"- Fausses alertes estimées : **{false_alerts:,}**.\n"
        f"- Coût d'investigation utilisé : **{money_usd(cost_inv)} par alerte**.\n"
        f"- Coût total d'investigation : **{money_usd(investigation_cost)}**.\n"
        f"- Bénéfice net estimé : **{money_usd(net)}**.\n\n"
        "Lecture métier : augmenter le recall protège mieux contre les pertes, mais augmente souvent la charge d'investigation. "
        "C'est exactement le compromis que le seuil métier doit piloter."
    )
    return response, ["Simulation métier", "Dataset brut", "Métriques modèle final"]


def answer_business_recommendation(question):
    q = normalize_text(question)
    if not any(word in q for word in ["recommande", "recommandation", "petite banque", "fausses alertes", "reduire", "choisir"]):
        return None

    if "petite banque" in q:
        answer = (
            "Pour une petite banque digitale, je recommanderais une stratégie prudente mais soutenable :\n\n"
            "- Garder un seuil proche de **0.56** comme point de départ, car il équilibre précision et recall.\n"
            "- Ne pas viser le recall maximal dès le départ, sinon les équipes risquent d'être saturées par les alertes.\n"
            "- Prioriser les alertes selon le montant, le produit C, les heures à risque et les signaux carte/device.\n"
            "- Mettre en place un suivi hebdomadaire des faux positifs pour ajuster progressivement le seuil.\n\n"
            "L'objectif n'est pas seulement de détecter plus, mais de détecter mieux avec une charge opérationnelle maîtrisée."
        )
    elif any(word in q for word in ["fausses alertes", "faux positifs", "reduire"]):
        answer = (
            "Pour réduire les fausses alertes, la banque peut agir sur quatre leviers :\n\n"
            "- Augmenter légèrement le seuil de décision pour privilégier la précision.\n"
            "- Prioriser les alertes à fort montant ou avec plusieurs signaux faibles combinés.\n"
            "- Créer une file de revue humaine uniquement pour les scores intermédiaires.\n"
            "- Réentraîner le modèle avec les retours des analystes fraude.\n\n"
            "Le seuil 0.56 est déjà un bon compromis, car il garde une précision proche de 88% tout en conservant un recall fort."
        )
    else:
        answer = (
            "La recommandation métier principale est de déployer le modèle comme un outil d'aide à la décision, pas comme un blocage automatique unique :\n\n"
            "- Scores élevés : blocage ou revue immédiate.\n"
            "- Scores moyens : vérification renforcée.\n"
            "- Scores faibles : autorisation avec monitoring.\n"
            "- Suivi continu du seuil selon le coût des fraudes, le coût d'investigation et l'expérience client.\n\n"
            "Cette approche est plus réaliste pour une banque digitale, car elle transforme les métriques ML en règles opérationnelles."
        )
    return answer, ["Stratégie métier", "Seuil métier", "KPIs modèle final"]


def answer_prediction_scope(question):
    q = normalize_text(question)
    if not any(word in q for word in ["predire", "prediction", "prédire", "prédiction", "tester", "scoring", "score xgboost"]):
        return None

    if any(word in q for word in ["complete", "complète", "xgboost", "vraie", "vrai"]):
        return (
            "Pour une **prédiction complète XGBoost**, il faut passer par la pipeline modèle complète : préparation des données, "
            "encodage des variables, construction des features et application du modèle entraîné. Le modèle final utilise environ "
            "**743 variables**, donc il ne peut pas être reconstruit fidèlement à partir d'une phrase courte.\n\n"
            "Ce que je peux faire ici :\n"
            "- analyser une transaction décrite en langage naturel avec un **score explicatif** ;\n"
            "- expliquer les facteurs de risque détectés ;\n"
            "- rappeler le seuil final **0.56** et les performances du modèle ;\n"
            "- orienter vers la page « Test du modèle » pour une simulation structurée.\n\n"
            "Exemple à me donner : `transaction de 800 USD produit C carte crédit à 7h`.",
            ["Pipeline modèle final", "Règles explicatives du dashboard"],
        )

    return (
        "Je peux t'aider à analyser une transaction, mais il faut distinguer deux niveaux :\n\n"
        "- **Score explicatif chatbot** : basé sur quelques signaux lisibles comme montant, produit, carte, heure, email et device.\n"
        "- **Prédiction complète XGBoost** : basée sur le jeu transformé complet avec environ 743 variables.\n\n"
        "Si tu veux une analyse conversationnelle, écris par exemple : "
        "`transaction de 800 USD produit C carte crédit à 7h`. "
        "Si tu veux une simulation structurée, utilise la page « Test du modèle ».",
        ["Règles explicatives du dashboard", "Pipeline modèle final"],
    )


def chatbot_answer(question):
    q = normalize_text(question)
    sources = load_chatbot_sources()

    if not q.strip():
        return "Pose-moi une question sur le modèle, les KPIs, le seuil, le SSL ou les résultats métier.", []

    if any(word in q for word in ["bonjour", "salut", "hello", "salam"]):
        return (
            "Bonjour. Je suis l'assistant d'explicabilité du projet. "
            "Je peux expliquer les performances, le seuil 0.56, les KPIs métier, le produit W, "
            "l'approche hybride SSL + XGBoost et l'impact financier.",
            [],
        )

    conversational_transaction = answer_conversational_transaction(question)
    if conversational_transaction:
        return conversational_transaction

    scenario_simulation = answer_scenario_simulation(question)
    if scenario_simulation:
        return scenario_simulation

    business_recommendation = answer_business_recommendation(question)
    if business_recommendation:
        return business_recommendation

    if "auc" in q and any(word in q for word in ["quoi", "veut", "signifie", "definition", "définition", "c'est quoi"]):
        return (
            "L'**AUC-ROC** mesure la capacité globale du modèle à classer les fraudes au-dessus des transactions légitimes. "
            "Une AUC proche de 1 signifie que le modèle sépare très bien les deux classes. "
            "Dans notre projet, l'AUC-ROC du modèle final est **0.9718**.\n\n"
            "Mais en fraude bancaire, la classe fraude est rare. C'est pourquoi on regarde aussi l'**Average Precision / PR-AUC**, "
            "plus adaptée aux données déséquilibrées. Ici, elle atteint **0.8608**, ce qui indique une bonne capacité à prioriser les alertes frauduleuses.",
            ["Explication KPI", "Modèle final"],
        )

    if "seuil" in q and any(word in q for word in ["quoi", "veut", "signifie", "definition", "définition", "c'est quoi"]):
        return (
            "Le **seuil de décision** est la limite à partir de laquelle une transaction est considérée comme suspecte. "
            "Par exemple, avec un seuil de **0.56**, une transaction dont le score de fraude est supérieur ou égal à 0.56 est classée comme fraude.\n\n"
            "Dans une banque digitale, ce seuil est un choix métier : un seuil plus bas détecte plus de fraudes mais crée plus de fausses alertes ; "
            "un seuil plus haut réduit les fausses alertes mais peut manquer des fraudes. Notre seuil final 0.56 équilibre precision, recall et charge d'investigation.",
            ["Seuil métier", "Explication KPI"],
        )

    if any(word in q for word in ["pas une prediction complete", "pas une prédiction complète", "prediction complete", "prédiction complète", "score explicatif"]):
        return (
            "Ce score n'est pas une prédiction complète XGBoost parce qu'il est calculé à partir de quelques signaux lisibles : "
            "montant, produit, type de carte, heure, email et device.\n\n"
            "Le modèle XGBoost final, lui, utilise le jeu de données transformé avec environ **743 variables** : variables transactionnelles, "
            "variables d'identité, encodages, agrégations et signaux issus de la préparation des données. Il produit une probabilité de fraude, "
            "puis le seuil **0.56** transforme cette probabilité en décision.\n\n"
            "Donc le chatbot sert à expliquer la logique métier d'une transaction de manière pédagogique. "
            "La prédiction complète doit rester dans la pipeline modèle, avec les mêmes transformations que pendant l'entraînement.",
            ["Règles explicatives du dashboard", "Pipeline modèle final"],
        )

    prediction_scope = answer_prediction_scope(question)
    if prediction_scope:
        return prediction_scope

    transaction_lookup = answer_transaction_lookup(question)
    if transaction_lookup:
        return transaction_lookup

    product_lookup = answer_product_lookup(question)
    if product_lookup:
        return product_lookup

    sample_analysis = answer_sample_analysis(question)
    if sample_analysis:
        return sample_analysis

    if any(word in q for word in ["ssl", "self supervised", "self-supervised"]):
        return (
            "Il faut valoriser le SSL comme **l'hypothèse scientifique initiale** du projet, pas comme le modèle final. "
            "L'idée de départ était d'exploiter la structure des transactions avec peu de labels frauduleux, ce qui est cohérent "
            "avec un contexte bancaire où la fraude est rare.\n\n"
            "Les expérimentations ont ensuite montré que, sur ce dataset tabulaire, **XGBoost optimisé** était plus performant "
            "pour la décision finale. C'est justement ce qui rend la démarche solide : le projet ne force pas le SSL à gagner, "
            "il montre une évolution expérimentale vers une approche hybride.\n\n"
            "Formulation soutenance : le SSL a servi à explorer l'apprentissage représentationnel, puis le système final combine "
            "XGBoost, seuil métier, KPIs et dashboard pour obtenir une solution plus exploitable par une banque digitale.",
            ["Comparaison modèles", "Labels limités", "Risque SSL guidé"],
        )

    validated_answers = [
        {
            "keywords": ["modele final", "modeles finales", "modèle final", "résultat final", "resultat final", "xgboost final"],
            "sources": ["Modèle final", "Hyperparamètres"],
            "answer": (
                "Le modèle final retenu est un **XGBoost optimisé** avec un seuil de décision **0.56**. "
                "Ses performances sur validation sont : AUC-ROC = 0.9718, Average Precision = 0.8608, "
                "F1-score = 0.8210, precision = 0.8797 et recall = 0.7697. "
                "Ce choix est cohérent avec une banque digitale : il détecte une part importante des fraudes "
                "tout en gardant une précision élevée pour limiter les fausses alertes."
            ),
        },
        {
            "keywords": ["problematique", "sujet", "question de recherche"],
            "sources": ["Guideline mémoire"],
            "answer": (
                f"La problématique retenue est : « {PROBLEMATIQUE} ». "
                "Elle valorise le SSL comme approche initiale, tout en assumant l'évolution vers une approche hybride "
                "avec XGBoost optimisé, seuil métier et explicabilité."
            ),
        },
        {
            "keywords": ["ssl", "self supervised", "hybride", "xgboost"],
            "sources": ["Comparaison modèles", "Labels limités", "Risque SSL guidé"],
            "answer": (
                "Le SSL est valorisé comme point de départ scientifique : il sert à explorer l'apprentissage de représentations "
                "à partir de transactions peu ou mal labellisées. Les expérimentations ont ensuite montré que, sur ce dataset tabulaire, "
                "XGBoost optimisé fournit les meilleures performances finales. Le système retenu est donc hybride : SSL pour la démarche "
                "expérimentale, XGBoost pour la décision, puis seuil métier et dashboard pour l'exploitation bancaire."
            ),
        },
        {
            "keywords": ["seuil", "0.56", "decision", "faux positifs", "investigation"],
            "sources": ["Seuil métier", "Modèle final"],
            "answer": (
                "Le seuil retenu est 0.56. Il maximise le F1-score du modèle final à 0.8210, avec une précision de 87.97% "
                "et un recall de 76.97%. Métierement, ce seuil garde un bon équilibre : détecter beaucoup de fraudes sans créer "
                "une charge excessive de fausses alertes pour les équipes d'investigation."
            ),
        },
        {
            "keywords": ["produit w", "product w", "volume", "produit"],
            "sources": ["KPIs métier"],
            "answer": (
                "Le produit W a le taux de fraude le plus faible, environ 2.04%, mais il concentre 439 670 transactions. "
                "Comme son volume est massif, il génère 8 969 fraudes en valeur absolue. La bonne lecture est donc : "
                "fraudes observées = volume de transactions × taux de fraude."
            ),
        },
        {
            "keywords": ["performance", "performances", "auc", "precision", "recall", "f1", "average precision", "pr auc", "metrique", "métrique"],
            "sources": ["Modèle final", "Hyperparamètres"],
            "answer": (
                "Le modèle final atteint AUC-ROC = 0.9718, Average Precision = 0.8608, F1-score = 0.8210, "
                "precision = 0.8797 et recall = 0.7697. Ces résultats montrent une forte capacité de classement, "
                "mais aussi une bonne performance sur la classe rare, ce qui est essentiel en détection de fraude."
            ),
        },
        {
            "keywords": ["auc veut dire", "c'est quoi auc", "que veut dire auc", "average precision", "pr auc"],
            "sources": ["Explication KPI"],
            "answer": (
                "L'AUC-ROC mesure la capacité globale du modèle à classer les fraudes au-dessus des transactions légitimes. "
                "Mais en fraude bancaire, la classe fraude est rare : l'Average Precision, ou PR-AUC, est donc encore plus parlante, "
                "car elle évalue la qualité du modèle sur la classe positive rare. Ici, l'Average Precision atteint 0.8608, "
                "ce qui montre une bonne capacité à prioriser les transactions suspectes."
            ),
        },
        {
            "keywords": ["carte credit", "carte crédit", "credit plus risque", "cartes credit", "cartes crédit"],
            "sources": ["KPIs métier"],
            "answer": (
                "Les cartes crédit apparaissent plus risquées dans l'analyse : environ 6.68% de fraude contre 2.43% pour les cartes débit. "
                "Métierement, cela peut s'expliquer par des usages plus exposés, des montants ou comportements différents, "
                "et une attractivité plus forte pour certains scénarios de fraude. Ce signal ne suffit pas seul, mais il devient utile "
                "quand il est combiné au produit, au montant, à l'heure, au device et à l'email."
            ),
        },
        {
            "keywords": ["impact", "benefice", "cout", "financier", "sauve", "alertes"],
            "sources": ["KPIs métier", "Modèle final", "Seuil métier"],
            "answer": (
                "Sur projection dataset complet, le modèle détecte environ 15 904 fraudes sur 20 663, avec 2 174 fausses alertes. "
                "Le montant sauvé estimé est de **2 373 635 USD**, pour un coût d'investigation de **271 170 USD**, "
                "soit un bénéfice net estimé à **2 102 465 USD**. "
                "C'est cette traduction financière qui donne du sens métier aux métriques ML."
            ),
        },
        {
            "keywords": ["hyperparametre", "tuning", "optimisation", "arbres", "profondeur"],
            "sources": ["Hyperparamètres"],
            "answer": (
                "Le tuning a retenu un XGBoost expressif mais régularisé : 562 arbres, profondeur 9, learning rate 0.149, "
                "subsample 0.822 et colsample_bytree 0.736. Cela permet de capter des combinaisons de signaux faibles "
                "tout en limitant le surapprentissage."
            ),
        },
        {
            "keywords": ["desequilibre", "rare", "classe", "labels", "labellisees"],
            "sources": ["KPIs métier"],
            "answer": (
                "Le dataset est fortement déséquilibré : 20 663 fraudes sur 590 540 transactions, soit 3.50%. "
                "Il y a environ une fraude pour 27 transactions légitimes. C'est pourquoi les métriques comme PR-AUC, recall, "
                "precision et F1 sont plus pertinentes qu'une simple accuracy."
            ),
        },
        {
            "keywords": ["chatbot", "assistant", "hallucination", "invente"],
            "sources": ["Guideline mémoire", "Rapports locaux"],
            "answer": (
                "Ce chatbot est conçu comme un assistant d'explicabilité. Il ne remplace pas le modèle de prédiction et ne répond "
                "pas librement hors périmètre. Ses réponses sont limitées aux rapports locaux du projet et aux résultats validés. "
                "Si une information n'est pas disponible, il doit le dire explicitement."
            ),
        },
    ]

    best = None
    best_score = 0
    for item in validated_answers:
        score = sum(1 for keyword in item["keywords"] if normalize_text(keyword) in q)
        if score > best_score:
            best = item
            best_score = score

    if best and best_score > 0:
        return best["answer"], best["sources"]

    excerpts = search_report_excerpts(question, sources)
    if excerpts and excerpts[0][0] >= 2:
        response_lines = [
            "Je n'ai pas de réponse dédiée pour cette formulation, mais voici les éléments fiables retrouvés dans les rapports :"
        ]
        used_sources = []
        for _, source_name, block in excerpts:
            clean_lines = []
            for line in block.splitlines():
                line = line.strip(" -=\t")
                if not line:
                    continue
                if len(line) > 180:
                    line = line[:180].rsplit(" ", 1)[0] + "..."
                clean_lines.append(line)
                if len(clean_lines) == 4:
                    break
            if clean_lines:
                response_lines.append(f"\n**{source_name}**")
                response_lines.extend(f"- {line}" for line in clean_lines)
            used_sources.append(source_name)
        return "\n".join(response_lines), used_sources

    return (
        "Je n'ai pas cette information dans les rapports du projet. "
        "Je peux répondre sur le seuil 0.56, les performances, le produit W, l'impact financier, "
        "les hyperparamètres, le déséquilibre des classes ou l'approche hybride SSL + XGBoost.",
        [],
    )


st.sidebar.title("🛡️ Navigation")
page = st.sidebar.radio("Navigation", [
    "🏠 Vue d'ensemble", "🔍 Exploration des données", "📈 KPIs Business",
    "🧪 Test du modèle", "📊 Performances", "💬 Assistant métier", "🔬 Itérations du projet",
], label_visibility="collapsed")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎯 Modèle final")
st.sidebar.metric("AUC-ROC", f"{FINAL_METRICS['auc']:.4f}")
st.sidebar.metric("F1-score", f"{FINAL_METRICS['f1']:.4f}")
st.sidebar.metric("Recall", f"{FINAL_METRICS['recall']:.1%}")
st.sidebar.metric("Precision", f"{FINAL_METRICS['precision']:.1%}")
st.sidebar.markdown("---")
st.sidebar.markdown("**Capstone PFE — Master Data science in business PSTB**")  
st.sidebar.caption("EL HAMDAOUI Mohamed · BEN ARFI Mahdi")
st.sidebar.caption("Encadrant : Mahdi ZARG AYOUNA")


if page == "🏠 Vue d'ensemble":
    st.title("🛡️ Détection de Fraude dans la Banque Digitale")
    st.markdown("### Self-Supervised Learning + XGBoost optimisé")
    st.markdown("---")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Transactions", "590 540")
    c2.metric("Taux de fraude", "3.50%")
    c3.metric("AUC-ROC", f"{FINAL_METRICS['auc']:.3f}", "+0.022")
    c4.metric("F1-score", f"{FINAL_METRICS['f1']:.3f}", "+0.155")
    c5.metric("Recall", f"{FINAL_METRICS['recall']:.1%}", "+16 pts")
    st.markdown("---")
    col1, col2 = st.columns([3, 2])
    with col1:
        st.subheader("📋 Problématique")
        st.info(f"**{PROBLEMATIQUE}**")
        st.markdown("""
        Démarche scientifique itérative :
        - Test du **SSL** (masquage de features + reconstruction)
        - Comparaison avec une **baseline XGBoost** à différents niveaux de labels
        - **Optimisation** du modèle final et **stratégie de seuil métier**
        - Traduction des performances en **impact financier concret**
        """)
    with col2:
        st.subheader("🎯 Taux de détection")
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=FINAL_METRICS["recall"]*100,
            number={"suffix": "%"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#2563eb"},
                   "steps": [{"range": [0, 50], "color": "#fee2e2"},
                             {"range": [50, 75], "color": "#fef3c7"},
                             {"range": [75, 100], "color": "#dcfce7"}]}))
        fig.update_layout(height=280)
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")
    st.subheader("🔄 Pipeline")
    cols = st.columns(6)
    steps = [("1. Extract", "Jointure IEEE-CIS"), ("2. Transform", "743 features"),
             ("3. SSL", "Pretraining"), ("4. XGBoost", "Classification"),
             ("5. Tuning", "Optimisation"), ("6. Décision", "Seuil métier")]
    for col, (t, d) in zip(cols, steps):
        col.markdown(f"**{t}**"); col.caption(d)


elif page == "🔍 Exploration des données":
    st.title("🔍 Exploration du dataset IEEE-CIS")
    try:
        df = load_raw_data()
        c1, c2, c3 = st.columns(3)
        c1.metric("Lignes", f"{len(df):,}")
        c2.metric("Colonnes", f"{df.shape[1]}")
        c3.metric("Fraudes", f"{df['isFraud'].sum():,}")
        st.dataframe(df.head(50), use_container_width=True, height=250)
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Distribution des classes")
            counts = df["isFraud"].value_counts()
            fig = px.pie(values=counts.values, names=["Légitime", "Fraude"],
                        color_discrete_sequence=["#2563eb", "#dc2626"], hole=0.4)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.subheader("Distribution des montants")
            max_amt = st.slider("Montant max", 100, 2000, 500, 50)
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=df[df.isFraud==0]["TransactionAmt"].clip(upper=max_amt),
                                       name="Légitime", marker_color="#2563eb", opacity=0.7))
            fig.add_trace(go.Histogram(x=df[df.isFraud==1]["TransactionAmt"].clip(upper=max_amt),
                                       name="Fraude", marker_color="#dc2626", opacity=0.7))
            fig.update_layout(barmode="overlay", height=350)
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Erreur : {e}")


elif page == "📈 KPIs Business":
    st.title("📈 KPIs Métier — Fraude Bancaire")
    try:
        df = load_raw_data()
        fraud = df[df.isFraud == 1]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Volume frauduleux", f"${fraud['TransactionAmt'].sum():,.0f}")
        c2.metric("Montant moyen fraude", f"${fraud['TransactionAmt'].mean():.0f}")
        c3.metric("Produit le + ciblé", "C (11.7%)")
        c4.metric("Pic de fraude", "7h (10.5%)")
        st.markdown("---")
        st.subheader("Taux de fraude par produit")
        prod = df.groupby("ProductCD").agg(total=("isFraud","count"), frauds=("isFraud","sum")).reset_index()
        prod["taux"] = prod["frauds"]/prod["total"]*100
        prod = prod.sort_values("taux", ascending=False)
        fig = px.bar(prod, x="ProductCD", y="taux", color="taux",
                    color_continuous_scale=["#22c55e","#eab308","#dc2626"],
                    text=[f"{t:.1f}%" for t in prod["taux"]])
        fig.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Effet volume : fraudes = transactions x taux")
        prod_exposure = prod.sort_values("total", ascending=False).copy()
        prod_exposure["bubble_size"] = 18 + prod_exposure["frauds"] / prod_exposure["frauds"].max() * 55
        fig = go.Figure(go.Scatter(
            x=prod_exposure["total"],
            y=prod_exposure["taux"],
            mode="markers+text",
            text=prod_exposure["ProductCD"],
            textposition="top center",
            marker=dict(
                size=prod_exposure["bubble_size"],
                color=prod_exposure["frauds"],
                colorscale=[[0, "#22c55e"], [0.55, "#eab308"], [1, "#dc2626"]],
                showscale=True,
                colorbar=dict(title="Fraudes"),
                line=dict(color="#111827", width=1),
            ),
            customdata=np.stack([prod_exposure["frauds"], prod_exposure["total"], prod_exposure["taux"]], axis=-1),
            hovertemplate=(
                "Produit %{text}<br>"
                "Transactions: %{customdata[1]:,.0f}<br>"
                "Taux fraude: %{customdata[2]:.2f}%<br>"
                "Fraudes: %{customdata[0]:,.0f}<extra></extra>"
            ),
        ))
        w_row = prod_exposure[prod_exposure["ProductCD"] == "W"].iloc[0]
        c_row = prod_exposure[prod_exposure["ProductCD"] == "C"].iloc[0]
        fig.add_annotation(x=w_row["total"], y=w_row["taux"], text="W: faible taux, tres fort volume",
                           showarrow=True, arrowhead=2, ax=-110, ay=-55)
        fig.add_annotation(x=c_row["total"], y=c_row["taux"], text="C: taux le plus eleve",
                           showarrow=True, arrowhead=2, ax=85, ay=35)
        fig.update_layout(
            height=440,
            title="Exposition transactionnelle vs risque relatif",
            xaxis_title="Nombre total de transactions",
            yaxis_title="Taux de fraude (%)",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.info("W genere beaucoup de fraudes car son volume de transactions est massif, meme si son taux de fraude est faible.")

        st.subheader("Distribution horaire des fraudes")
        df["hour"] = (df["TransactionDT"]/3600%24).astype(int)
        hourly = df.groupby("hour").agg(total=("isFraud","count"), frauds=("isFraud","sum")).reset_index()
        hourly["taux"] = hourly["frauds"]/hourly["total"]*100
        fig = px.bar(hourly, x="hour", y="taux", color="taux",
                    color_continuous_scale=["#22c55e","#eab308","#dc2626"])
        fig.update_layout(height=350, showlegend=False, xaxis=dict(dtick=1))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("Graphes explicatifs issus des rapports métier")
        st.caption(
            "Ces figures reprennent les analyses générées dans reports/figures. "
            "Elles servent à expliquer le risque bancaire sous trois angles : exposition, comportement et impact financier."
        )

        business_figures = [
            (
                "business_produits_exposition.png",
                "Exposition produit : pourquoi W génère beaucoup de fraudes",
                "W a un faible taux de fraude, mais un très grand volume de transactions. Le volume explique donc son nombre élevé de fraudes en valeur absolue.",
            ),
            (
                "business_montants.png",
                "Montants : distribution et valeur financière exposée",
                "Ce graphe montre si les fraudes se concentrent sur certains niveaux de montant et aide à estimer l'impact financier potentiel.",
            ),
            (
                "business_temporel.png",
                "Temporalité : heures où le risque augmente",
                "La lecture par heure permet de relier le modèle à une logique opérationnelle : surveillance renforcée sur les créneaux les plus risqués.",
            ),
            (
                "business_emails.png",
                "Domaines email : signaux de risque client",
                "Certains domaines concentrent plus de fraudes et peuvent devenir des variables utiles pour le scoring.",
            ),
            (
                "business_cartes.png",
                "Carte et réseau : différence de risque selon les moyens de paiement",
                "Le type de carte et le réseau apportent une lecture métier du risque transactionnel.",
            ),
            (
                "business_devices.png",
                "Device : canal utilisé par le client",
                "Le canal desktop/mobile aide à comprendre les contextes où les fraudes apparaissent davantage.",
            ),
            (
                "base_threshold_tuning.png",
                "Seuil de décision : compromis performance et coût métier",
                "Le seuil 0.56 retenu garde un bon équilibre entre détection, précision et charge d'investigation.",
            ),
            (
                "business_desequilibre.png",
                "Déséquilibre des classes : justification du problème",
                "La fraude est rare par rapport aux transactions légitimes, ce qui justifie l'usage de métriques adaptées comme PR-AUC, recall et F1.",
            ),
        ]

        for idx in range(0, len(business_figures), 2):
            cols = st.columns(2)
            for col, (filename, title, explanation) in zip(cols, business_figures[idx:idx + 2]):
                fig_path = os.path.join(FIGURES_DIR, filename)
                with col:
                    st.markdown(f"#### {title}")
                    if os.path.exists(fig_path):
                        st.image(fig_path, use_container_width=True)
                    else:
                        st.warning(f"Figure introuvable : {fig_path}")
                    st.caption(explanation)

        st.markdown("---")
        st.subheader("💰 Simulateur d'impact financier")
        col1, col2 = st.columns([1,2])
        with col1:
            recall_pct = st.slider("Recall (%)", 10, 95, 77)
            cost_inv = st.number_input("Coût investigation ($)", 5, 50, 15)
        recall = recall_pct/100
        n_det = int(len(fraud)*recall)
        saved = fraud["TransactionAmt"].sum()*recall
        n_fp = int(n_det*(1-FINAL_METRICS["precision"])/FINAL_METRICS["precision"])
        net = saved-(n_det+n_fp)*cost_inv
        with col2:
            cc1,cc2,cc3 = st.columns(3)
            cc1.metric("Fraudes détectées", f"{n_det:,}")
            cc2.metric("Montant sauvé", f"${saved:,.0f}")
            cc3.metric("Bénéfice net", f"${net:,.0f}")
    except Exception as e:
        st.error(f"Erreur : {e}")


elif page == "🧪 Test du modèle":
    st.title("🧪 Tester une transaction")
    st.markdown("Simulez une transaction pour obtenir un score de risque.")
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### 💳 Transaction")
        amount = st.number_input("Montant ($)", 0.0, 50000.0, 120.0, 10.0)
        product = st.selectbox("Produit", ["W","H","C","S","R"])
        card_type = st.selectbox("Type de carte", ["credit","debit"])
        card_net = st.selectbox("Réseau", ["visa","mastercard","discover","american express"])
    with col2:
        st.markdown("#### 🌍 Contexte")
        hour = st.slider("Heure", 0, 23, 14)
        email = st.selectbox("Email", ["gmail.com","yahoo.com","hotmail.com","outlook.com","mail.com","icloud.com","autre"])
        device = st.selectbox("Device", ["desktop","mobile","inconnu"])
    with col3:
        st.markdown("#### Signaux calcules")
        is_night = 1 if (hour >= 22 or hour <= 6) else 0
        is_peak = 1 if (5 <= hour <= 10) else 0
        is_round_amount = 1 if amount % 1 == 0 else 0

        signal_rows = pd.DataFrame({
            "Signal": ["Transaction de nuit", "Heure pic fraude", "Montant rond"],
            "Base de calcul": [
                "Heure >= 22h ou <= 6h",
                "Heure entre 5h et 10h",
                "Montant sans centimes",
            ],
            "Valeur saisie": [f"{hour}h", f"{hour}h", f"${amount:,.2f}"],
            "Statut": [
                "Oui" if is_night else "Non",
                "Oui" if is_peak else "Non",
                "Oui" if is_round_amount else "Non",
            ],
        })
        st.dataframe(signal_rows, use_container_width=True, hide_index=True)
        st.caption("Ces indicateurs sont des variables derivees calculees uniquement a partir des champs saisis, avant prediction du modele.")
    st.markdown("---")
    if st.button("🔍 Analyser", type="primary", use_container_width=True):
        risk = 0; factors = []
        prod_risk = {"C":11.7,"S":5.9,"H":4.8,"R":3.8,"W":2.0}
        if prod_risk.get(product,3.5) > 5:
            risk += 25; factors.append(f"Produit {product} à haut risque ({prod_risk[product]:.1f}%)")
        elif prod_risk.get(product,3.5) > 3.5:
            risk += 10
        if is_peak:
            risk += 20; factors.append(f"Heure {hour}h dans le pic de fraude")
        if is_night:
            risk += 8; factors.append("Transaction nocturne")
        email_risk = {"mail.com":19.0,"outlook.com":9.5,"hotmail.com":5.3,"gmail.com":4.4}
        er = email_risk.get(email,3.0)
        if er > 8:
            risk += 25; factors.append(f"Email {email} très risqué ({er:.1f}%)")
        elif er > 5:
            risk += 10; factors.append(f"Email {email} risqué ({er:.1f}%)")
        if device == "mobile":
            risk += 15; factors.append("Mobile (10.2% vs 6.5% desktop)")
        if card_type == "credit":
            risk += 10; factors.append("Carte crédit (6.7% vs 2.4% débit)")
        if card_net == "discover":
            risk += 10; factors.append("Réseau Discover (7.7%)")
        if amount > 500:
            risk += 8; factors.append(f"Montant élevé (${amount:.0f})")
        risk = min(risk, 100)
        st.markdown("---")
        col1, col2 = st.columns([1,2])
        with col1:
            if risk >= 70:
                st.error(f"## 🚨 Risque : {risk}/100")
                st.error("**SUSPECTE** — Blocage recommandé")
            elif risk >= 40:
                st.warning(f"## ⚠️ Risque : {risk}/100")
                st.warning("**SURVEILLANCE** — Vérification")
            else:
                st.success(f"## ✅ Risque : {risk}/100")
                st.success("**NORMALE** — Autorisation")
        with col2:
            st.markdown("#### Facteurs détectés")
            if factors:
                for f in factors:
                    st.markdown(f"- ⚠️ {f}")
            else:
                st.markdown("Aucun facteur de risque majeur.")
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=risk,
            gauge={"axis":{"range":[0,100]},
                   "bar":{"color":"#dc2626" if risk>=70 else "#eab308" if risk>=40 else "#22c55e"},
                   "steps":[{"range":[0,40],"color":"#dcfce7"},
                            {"range":[40,70],"color":"#fef3c7"},
                            {"range":[70,100],"color":"#fee2e2"}]}))
        fig.update_layout(height=250)
        st.plotly_chart(fig, use_container_width=True)


elif page == "📊 Performances":
    st.title("📊 Performances du modèle final")
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("AUC-ROC", f"{FINAL_METRICS['auc']:.4f}")
    c2.metric("Avg Precision", f"{FINAL_METRICS['avg_precision']:.4f}")
    c3.metric("F1-score", f"{FINAL_METRICS['f1']:.4f}")
    c4.metric("Precision", f"{FINAL_METRICS['precision']:.4f}")
    c5.metric("Recall", f"{FINAL_METRICS['recall']:.4f}")
    st.markdown("---")
    y_true, y_proba = load_predictions()
    if y_true is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Courbe ROC")
            from sklearn.metrics import roc_curve
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                     name=f"AUC={FINAL_METRICS['auc']:.4f}",
                                     line=dict(color="#2563eb", width=3)))
            fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                                     line=dict(color="gray", dash="dash")))
            fig.update_layout(height=400, xaxis_title="FPR", yaxis_title="TPR")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.subheader("Matrice de confusion")
            from sklearn.metrics import confusion_matrix
            y_pred = (y_proba >= FINAL_METRICS["threshold"]).astype(int)
            cm = confusion_matrix(y_true, y_pred)
            fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                           x=["Légitime","Fraude"], y=["Légitime","Fraude"],
                           labels=dict(x="Prédit", y="Réel"))
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        st.markdown("---")
        st.subheader("🎚️ Impact du seuil de décision")
        seuil = st.slider("Seuil", 0.05, 0.95, FINAL_METRICS["threshold"], 0.01)
        from sklearn.metrics import precision_score, recall_score, f1_score
        y_pred_s = (y_proba >= seuil).astype(int)
        cc1,cc2,cc3,cc4 = st.columns(4)
        cc1.metric("Precision", f"{precision_score(y_true,y_pred_s,zero_division=0):.3f}")
        cc2.metric("Recall", f"{recall_score(y_true,y_pred_s,zero_division=0):.3f}")
        cc3.metric("F1", f"{f1_score(y_true,y_pred_s,zero_division=0):.3f}")
        cc4.metric("Fraudes détectées", f"{int(((y_pred_s==1)&(y_true==1)).sum())}")
    else:
        st.warning("Lancez hyperparameter_tuning.py pour générer tuned_predictions.npz")
    st.markdown("---")
    st.subheader("⚙️ Hyperparamètres optimaux")
    st.caption(
        "Lecture métier : ces paramètres définissent comment XGBoost apprend les profils de fraude "
        "sans sur-réagir aux transactions rares ou atypiques."
    )

    p1, p2, p3 = st.columns(3)
    p1.metric("Complexité contrôlée", f"Profondeur {BEST_PARAMS['max_depth']}")
    p2.metric("Nombre d'arbres", f"{BEST_PARAMS['n_estimators']}")
    p3.metric("Vitesse d'apprentissage", f"{BEST_PARAMS['learning_rate']:.3f}")

    st.info(
        "Le modèle final est volontairement expressif : profondeur 9 et 562 arbres pour capter "
        "des combinaisons de signaux faibles. La régularisation et l'échantillonnage évitent "
        "qu'il mémorise trop les cas particuliers."
    )

    tuning_readable = pd.DataFrame([
        {
            "Paramètre": "max_depth",
            "Valeur retenue": BEST_PARAMS["max_depth"],
            "Rôle simple": "Profondeur des arbres",
            "Lecture fraude bancaire": "Permet de combiner plusieurs signaux : produit, heure, montant, device, email.",
        },
        {
            "Paramètre": "n_estimators",
            "Valeur retenue": BEST_PARAMS["n_estimators"],
            "Rôle simple": "Nombre d'arbres",
            "Lecture fraude bancaire": "Accumule beaucoup de petites règles pour mieux distinguer fraude et transaction normale.",
        },
        {
            "Paramètre": "learning_rate",
            "Valeur retenue": f"{BEST_PARAMS['learning_rate']:.3f}",
            "Rôle simple": "Rythme d'apprentissage",
            "Lecture fraude bancaire": "Apprentissage progressif pour améliorer la précision sans devenir instable.",
        },
        {
            "Paramètre": "subsample",
            "Valeur retenue": f"{BEST_PARAMS['subsample']:.3f}",
            "Rôle simple": "Part des lignes utilisées par arbre",
            "Lecture fraude bancaire": "Réduit le risque de surapprentissage sur quelques transactions atypiques.",
        },
        {
            "Paramètre": "colsample_bytree",
            "Valeur retenue": f"{BEST_PARAMS['colsample_bytree']:.3f}",
            "Rôle simple": "Part des variables utilisées par arbre",
            "Lecture fraude bancaire": "Force le modèle à ne pas dépendre d'un seul indicateur de fraude.",
        },
        {
            "Paramètre": "reg_alpha / reg_lambda",
            "Valeur retenue": f"{BEST_PARAMS['reg_alpha']:.3f} / {BEST_PARAMS['reg_lambda']:.3f}",
            "Rôle simple": "Régularisation",
            "Lecture fraude bancaire": "Stabilise les décisions et limite les faux positifs inutiles.",
        },
        {
            "Paramètre": "min_child_weight",
            "Valeur retenue": BEST_PARAMS["min_child_weight"],
            "Rôle simple": "Seuil minimal pour créer une règle",
            "Lecture fraude bancaire": "Empêche de créer des règles trop fragiles sur très peu d'exemples.",
        },
        {
            "Paramètre": "gamma",
            "Valeur retenue": f"{BEST_PARAMS['gamma']:.3f}",
            "Rôle simple": "Gain minimal pour diviser un arbre",
            "Lecture fraude bancaire": "Une nouvelle règle est acceptée seulement si elle apporte un vrai gain.",
        },
    ])
    st.dataframe(tuning_readable, use_container_width=True, hide_index=True)

    st.success(
        "Conclusion : le tuning a trouvé un modèle plus sensible aux fraudes tout en gardant une précision élevée. "
        "C'est cohérent avec une banque digitale : détecter plus, mais sans saturer les équipes avec trop de fausses alertes."
    )


elif page == "💬 Assistant métier":
    st.title("💬 Assistant métier fraude bancaire")
    st.caption(
        "Assistant local d'explicabilité : il répond uniquement à partir des résultats validés du projet "
        "et des rapports présents dans reports/results."
    )

    st.info(
        "Garde-fou : cet assistant n'est pas un modèle de prédiction et ne remplace pas la page « Test du modèle ». "
        "S'il ne trouve pas l'information dans le périmètre du projet, il doit répondre qu'il ne sait pas."
    )

    st.markdown("#### Les 5 piliers de l'assistant")
    pillars = pd.DataFrame([
        {
            "Pilier": "Analyse de transaction",
            "Ce que l'utilisateur peut demander": "transaction de 800$ produit C carte crédit à 7h",
            "Réponse attendue": "Score de risque, décision suggérée et facteurs détectés",
        },
        {
            "Pilier": "Résultats du projet",
            "Ce que l'utilisateur peut demander": "le modèle final, quel est le recall ?",
            "Réponse attendue": "Métriques validées : AUC, PR-AUC, F1, précision, recall, seuil",
        },
        {
            "Pilier": "Explication des KPIs",
            "Ce que l'utilisateur peut demander": "que veut dire AUC ? pourquoi carte crédit ?",
            "Réponse attendue": "Vulgarisation technique orientée banque digitale",
        },
        {
            "Pilier": "Recommandations métier",
            "Ce que l'utilisateur peut demander": "comment réduire les fausses alertes ?",
            "Réponse attendue": "Conseils de seuil, priorisation et organisation opérationnelle",
        },
        {
            "Pilier": "Simulation de scénarios",
            "Ce que l'utilisateur peut demander": "si on détecte 90% des fraudes ?",
            "Réponse attendue": "Impact estimé : fraudes détectées, coût, bénéfice net",
        },
    ])
    st.dataframe(pillars, use_container_width=True, hide_index=True)

    sources = load_chatbot_sources()
    with st.expander("Sources locales utilisées par l'assistant"):
        if sources:
            for source_name in sources:
                st.markdown(f"- {source_name}")
        else:
            st.warning("Aucun rapport texte n'a été trouvé dans reports/results.")

    if "business_chat_history" not in st.session_state:
        st.session_state.business_chat_history = [
            {
                "role": "assistant",
                "content": (
                    "Bonjour. Je peux expliquer le seuil 0.56, les performances du modèle final, "
                    "l'impact financier, le produit W, l'approche hybride SSL + XGBoost et les KPIs métier."
                ),
                "sources": [],
            }
        ]

    st.markdown("#### Questions utiles pour la soutenance")
    suggestions = [
        "Transaction de 800$ produit C carte crédit à 7h du matin.",
        "Si on détecte 90% des fraudes, quel impact ?",
        "Comment réduire les fausses alertes ?",
        "Le modèle final.",
        "Pourquoi le seuil 0.56 a-t-il été retenu ?",
        "Pourquoi le produit W génère beaucoup de fraudes ?",
        "Analyse le produit C.",
        "Affiche la transaction 2987000.",
        "Analyse un échantillon de transactions.",
        "Comment valoriser le SSL alors que XGBoost est le modèle final ?",
        "Quel est l'impact financier du modèle ?",
        "Pourquoi le déséquilibre des classes est important ?",
        "Explique les performances du modèle final.",
    ]
    selected_question = None
    cols = st.columns(3)
    for idx, suggestion in enumerate(suggestions):
        if cols[idx % 3].button(suggestion, key=f"suggestion_{idx}", use_container_width=True):
            selected_question = suggestion

    col_clear, _ = st.columns([1, 3])
    if col_clear.button("Effacer la conversation", use_container_width=True):
        st.session_state.business_chat_history = st.session_state.business_chat_history[:1]

    user_question = st.chat_input("Ex: transaction de 800$ produit C carte crédit à 7h, seuil 0.56, modèle final, scénario 90%...")
    question_to_process = user_question or selected_question
    if question_to_process:
        st.session_state.business_chat_history.append({
            "role": "user",
            "content": question_to_process,
            "sources": [],
        })
        answer, answer_sources = chatbot_answer(question_to_process)
        st.session_state.business_chat_history.append({
            "role": "assistant",
            "content": answer,
            "sources": answer_sources,
        })

    st.markdown("---")
    for message in st.session_state.business_chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                st.caption("Sources : " + ", ".join(message["sources"]))


elif page == "🔬 Itérations du projet":
    st.title("🔬 Démarche itérative")
    st.markdown("Chaque itération a amélioré le modèle.")
    st.markdown("---")
    iterations = pd.DataFrame({
        "Itération": ["MLP V1 (gelé)","MLP V2 (dégelé)","XGBoost base",
                      "XGBoost + adaptatif","XGBoost optimisé (final)"],
        "AUC": [0.8224, 0.8932, 0.9494, 0.9494, 0.9718],
        "Precision": [0.7910, 0.5047, 0.7340, 0.7340, 0.8797],
        "Recall": [0.1365, 0.4319, 0.6100, 0.6660, 0.7697],
        "F1": [0.2328, 0.4654, 0.6660, 0.6660, 0.8210],
    })
    st.dataframe(
        iterations.style.background_gradient(subset=["AUC","Recall","F1"], cmap="Greens")
                       .format({"AUC":"{:.4f}","Precision":"{:.4f}","Recall":"{:.4f}","F1":"{:.4f}"}),
        use_container_width=True, hide_index=True)
    st.markdown("---")
    fig = make_subplots(rows=1, cols=3, subplot_titles=["AUC-ROC","Recall","F1-score"])
    for i, metric in enumerate(["AUC","Recall","F1"]):
        fig.add_trace(go.Scatter(x=iterations["Itération"], y=iterations[metric],
                                mode="lines+markers", line=dict(width=3),
                                marker=dict(size=10), showlegend=False), row=1, col=i+1)
    fig.update_layout(height=400); fig.update_xaxes(tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")
    st.subheader("📝 Conclusions clés pour le mémoire")
    st.markdown("""
    1. **Le SSL n'a pas surpassé XGBoost** sur ce dataset tabulaire — résultat honnête,
       cohérent avec la littérature (SSL excelle en image/texte, moins en tabulaire).
    2. **L'optimisation des hyperparamètres a été décisive** : +15 pts de F1, de 61% à 77% de recall.
    3. **Les fraudes non détectées sont sophistiquées** : elles imitent les transactions légitimes.
    4. **La stratégie de seuil est un choix métier** : recall (sécurité) vs precision (efficacité).
    """)
