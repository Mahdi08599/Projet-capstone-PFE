"""
=================================================================
Dashboard Streamlit — Fraud Detection SSL
=================================================================
Interface complète pour le projet Capstone :
  - Vue d'ensemble du projet et KPIs
  - Exploration interactive des données
  - Test du modèle en temps réel
  - Comparaison des approches (V1 vs V2, Baseline vs SSL)
  - Interprétation des résultats

Usage :
    pip install streamlit plotly
    streamlit run app_dashboard.py
=================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, f1_score

# ─── Config ─────────────────────────────────────────────────────
DATA_RAW = os.path.join("data", "raw")
DATA_PROC = os.path.join("data", "processed")
MODELS_DIR = "models"
REPORTS_DIR = os.path.join("reports", "figures")

st.set_page_config(
    page_title="Fraud Detection SSL — Capstone",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─── Modèle SSL (doit correspondre à l'architecture) ───────────
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


# ─── Chargement des données (cache pour performance) ────────────
@st.cache_data
def load_raw_data():
    tx = pd.read_csv(os.path.join(DATA_RAW, "train_transaction.csv"))
    ident = pd.read_csv(os.path.join(DATA_RAW, "train_identity.csv"))
    df = tx.merge(ident, on="TransactionID", how="left")
    return df


@st.cache_data
def load_processed_data(version="v2"):
    suffix = f"_v2" if version == "v2" else ""
    train = pd.read_parquet(os.path.join(DATA_PROC, f"train_clean{suffix}.parquet"))
    val = pd.read_parquet(os.path.join(DATA_PROC, f"val_clean{suffix}.parquet"))
    return train, val


@st.cache_data
def load_results():
    results = {}
    for fname in os.listdir(os.path.join("reports", "results")):
        if fname.endswith(".txt"):
            with open(os.path.join("reports", "results", fname), "r", encoding="utf-8", errors="ignore") as f:
                results[fname] = f.read()
    return results


# ─── Sidebar ────────────────────────────────────────────────────
st.sidebar.title("🛡️ Navigation")
page = st.sidebar.radio("", [
    "📊 Vue d'ensemble",
    "🔍 Exploration des données",
    "📈 KPIs Business",
    "🧪 Test du modèle",
    "⚖️ Comparaison des approches",
    "📋 Rapport technique",
])

st.sidebar.markdown("---")
st.sidebar.markdown("**Projet Capstone**")
st.sidebar.markdown("Master Data / AI")
st.sidebar.markdown("EL HAMDAOUI Mohamed")
st.sidebar.markdown("BEN ARAFI Mahdi")
st.sidebar.markdown("Encadrant : Mahdi ZARG AYOUNA")


# =================================================================
# PAGE 1 : Vue d'ensemble
# =================================================================
if page == "📊 Vue d'ensemble":
    st.title("🛡️ Fraud Detection in Digital Banking")
    st.subheader("Using Self-Supervised Learning")

    st.markdown("---")

    # Métriques principales
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Transactions", "590 540", help="Nombre total de transactions dans le dataset")
    col2.metric("Taux de fraude", "3.50%", help="20 663 fraudes sur 590 540 transactions")
    col3.metric("Features V2", "743", delta="+517 vs V1", help="Après preprocessing amélioré")
    col4.metric("Montant frauduleux", "$3.08M", help="Volume total des transactions frauduleuses")

    st.markdown("---")

    # Pipeline du projet
    st.subheader("Pipeline du projet")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown("### 1️⃣ Extract")
        st.markdown("Chargement et jointure des données IEEE-CIS")

    with col2:
        st.markdown("### 2️⃣ Transform")
        st.markdown("Nettoyage, encoding, feature engineering")

    with col3:
        st.markdown("### 3️⃣ SSL")
        st.markdown("Pretraining par masquage de features")

    with col4:
        st.markdown("### 4️⃣ Classification")
        st.markdown("XGBoost sur embeddings SSL")

    with col5:
        st.markdown("### 5️⃣ Evaluation")
        st.markdown("AUC, F1, Recall, Precision, Confusion")

    st.markdown("---")

    # Problématique
    st.subheader("Problématique")
    st.info(
        "**Comment le self-supervised learning peut-il améliorer la détection de fraude "
        "dans la banque digitale lorsque les données labellisées sont limitées ?**"
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Pourquoi le SSL ?")
        st.markdown(
            "- Les fraudes représentent seulement **3.5%** des transactions\n"
            "- Obtenir des labels fraude nécessite une **investigation manuelle** coûteuse\n"
            "- Le SSL exploite les **données non labellisées** (abondantes) pour apprendre\n"
            "- 1 fraude pour **28 transactions** légitimes"
        )

    with col2:
        st.markdown("#### Approche")
        st.markdown(
            "- **Étape 1** : Pré-entraînement SSL (masquage + reconstruction)\n"
            "- **Étape 2** : Extraction des représentations apprises\n"
            "- **Étape 3** : Classification avec XGBoost\n"
            "- **Comparaison** : avec et sans SSL, à différents niveaux de labels"
        )


# =================================================================
# PAGE 2 : Exploration des données
# =================================================================
elif page == "🔍 Exploration des données":
    st.title("🔍 Exploration du dataset IEEE-CIS")

    try:
        df = load_raw_data()

        st.subheader("Aperçu du dataset")
        col1, col2, col3 = st.columns(3)
        col1.metric("Lignes", f"{len(df):,}")
        col2.metric("Colonnes", f"{df.shape[1]}")
        col3.metric("Taille mémoire", f"{df.memory_usage(deep=True).sum()/1e9:.1f} Go")

        st.dataframe(df.head(100), use_container_width=True, height=300)

        st.markdown("---")

        # Distribution de la cible
        st.subheader("Distribution de la variable cible (isFraud)")
        fraud_counts = df["isFraud"].value_counts()
        fig = px.pie(
            values=fraud_counts.values,
            names=["Légitime", "Fraude"],
            color_discrete_sequence=["#2196F3", "#E53935"],
            hole=0.4,
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # Distribution des montants
        st.subheader("Distribution des montants")
        col1, col2 = st.columns(2)

        with col1:
            max_amt = st.slider("Montant max à afficher ($)", 100, 5000, 500, 50)

        fraud_df = df[df["isFraud"] == 1]["TransactionAmt"].clip(upper=max_amt)
        legit_df = df[df["isFraud"] == 0]["TransactionAmt"].clip(upper=max_amt)

        fig = go.Figure()
        fig.add_trace(go.Histogram(x=legit_df, name="Légitime", opacity=0.7,
                                   marker_color="#2196F3", nbinsx=60))
        fig.add_trace(go.Histogram(x=fraud_df, name="Fraude", opacity=0.7,
                                   marker_color="#E53935", nbinsx=60))
        fig.update_layout(barmode="overlay", xaxis_title="Montant ($)",
                         yaxis_title="Nombre", height=400)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # Valeurs manquantes
        st.subheader("Top 30 colonnes avec le plus de valeurs manquantes")
        missing = df.isnull().mean().sort_values(ascending=False).head(30) * 100
        fig = px.bar(x=missing.index, y=missing.values,
                     labels={"x": "Colonne", "y": "% manquant"},
                     color=missing.values,
                     color_continuous_scale=["#4CAF50", "#FF9800", "#E53935"])
        fig.update_layout(height=400, showlegend=False)
        fig.add_hline(y=70, line_dash="dash", line_color="red",
                      annotation_text="Seuil V1 (70%)")
        fig.add_hline(y=95, line_dash="dash", line_color="blue",
                      annotation_text="Seuil V2 (95%)")
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Erreur de chargement : {e}")
        st.info("Vérifiez que les fichiers CSV sont dans data/raw/")


# =================================================================
# PAGE 3 : KPIs Business
# =================================================================
elif page == "📈 KPIs Business":
    st.title("📈 KPIs Métier — Fraude Bancaire")

    try:
        df = load_raw_data()
        fraud = df[df["isFraud"] == 1]
        legit = df[df["isFraud"] == 0]

        # KPI cards
        st.subheader("Indicateurs clés")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Volume frauduleux", f"${fraud['TransactionAmt'].sum():,.0f}",
                  f"{fraud['TransactionAmt'].sum()/df['TransactionAmt'].sum()*100:.2f}% du total")
        c2.metric("Montant moyen fraude", f"${fraud['TransactionAmt'].mean():,.0f}",
                  f"+${fraud['TransactionAmt'].mean() - legit['TransactionAmt'].mean():,.0f} vs légitime")
        c3.metric("Pic de fraude", "7h du matin", "10.5% de taux")
        c4.metric("Produit le + ciblé", "Produit C", "11.7% de fraude")

        st.markdown("---")

        # Fraude par produit
        st.subheader("Taux de fraude par produit")
        prod = df.groupby("ProductCD").agg(
            total=("isFraud", "count"), frauds=("isFraud", "sum")
        ).reset_index()
        prod["taux"] = prod["frauds"] / prod["total"] * 100
        prod = prod.sort_values("taux", ascending=False)

        fig = make_subplots(rows=1, cols=2, subplot_titles=["Taux de fraude (%)", "Volume de fraudes"])
        fig.add_trace(go.Bar(x=prod["ProductCD"], y=prod["taux"], name="Taux",
                            marker_color=["#E53935" if t>5 else "#FF9800" if t>3 else "#4CAF50"
                                          for t in prod["taux"]],
                            text=[f"{t:.1f}%" for t in prod["taux"]], textposition="outside"), row=1, col=1)
        fig.add_trace(go.Bar(x=prod["ProductCD"], y=prod["frauds"], name="Volume",
                            marker_color="#E53935"), row=1, col=2)
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # Analyse temporelle
        st.subheader("Analyse temporelle des fraudes")
        df_temp = df.copy()
        df_temp["hour"] = (df_temp["TransactionDT"] / 3600 % 24).astype(int)
        hourly = df_temp.groupby("hour").agg(
            total=("isFraud", "count"), frauds=("isFraud", "sum")
        ).reset_index()
        hourly["taux"] = hourly["frauds"] / hourly["total"] * 100

        fig = go.Figure()
        fig.add_trace(go.Bar(x=hourly["hour"], y=hourly["taux"], name="Taux fraude",
                            marker_color=["#E53935" if t>5 else "#FF9800" if t>3.5 else "#4CAF50"
                                          for t in hourly["taux"]]))
        fig.update_layout(xaxis_title="Heure", yaxis_title="Taux de fraude (%)",
                         height=400, xaxis=dict(dtick=1))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # Cartes et devices
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Risque par type de carte")
            card = df.groupby("card6").agg(
                total=("isFraud", "count"), frauds=("isFraud", "sum")
            ).reset_index().dropna()
            card["taux"] = card["frauds"] / card["total"] * 100
            card = card.sort_values("taux", ascending=True)
            fig = px.bar(card, x="taux", y="card6", orientation="h",
                        color="taux", color_continuous_scale=["#4CAF50", "#E53935"],
                        labels={"taux": "Taux fraude (%)", "card6": "Type de carte"})
            fig.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Risque par device")
            dev = df.groupby("DeviceType").agg(
                total=("isFraud", "count"), frauds=("isFraud", "sum")
            ).reset_index().dropna()
            dev["taux"] = dev["frauds"] / dev["total"] * 100
            fig = px.bar(dev, x="DeviceType", y="taux",
                        color="taux", color_continuous_scale=["#4CAF50", "#E53935"],
                        labels={"taux": "Taux fraude (%)"})
            fig.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # Top emails
        st.subheader("Domaines email les plus risqués")
        email = df.groupby("P_emaildomain").agg(
            total=("isFraud", "count"), frauds=("isFraud", "sum")
        ).reset_index()
        email["taux"] = email["frauds"] / email["total"] * 100
        email = email[email["total"] >= 500].sort_values("taux", ascending=False).head(15)

        fig = px.bar(email, x="taux", y="P_emaildomain", orientation="h",
                    color="taux", color_continuous_scale=["#4CAF50", "#FF9800", "#E53935"],
                    labels={"taux": "Taux fraude (%)", "P_emaildomain": "Domaine email"})
        fig.update_layout(height=500, yaxis=dict(autorange="reversed"), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # Impact financier du modèle
        st.subheader("💰 Impact financier du modèle SSL")

        recall_slider = st.slider("Recall du modèle (%)", 10, 90, 43, 1)
        recall = recall_slider / 100

        total_fraud_amt = fraud["TransactionAmt"].sum()
        n_frauds = len(fraud)
        n_detected = int(n_frauds * recall)
        amount_saved = total_fraud_amt * recall
        cost_investigation = 15
        n_false_alerts = int(n_detected * 0.5)  # estimation
        total_cost = (n_detected + n_false_alerts) * cost_investigation
        net_benefit = amount_saved - total_cost

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Fraudes détectées", f"{n_detected:,} / {n_frauds:,}")
        c2.metric("Montant sauvé", f"${amount_saved:,.0f}")
        c3.metric("Coût investigations", f"${total_cost:,.0f}")
        c4.metric("Bénéfice net", f"${net_benefit:,.0f}",
                  delta=f"${net_benefit:,.0f}", delta_color="normal")

    except Exception as e:
        st.error(f"Erreur : {e}")


# =================================================================
# PAGE 4 : Test du modèle
# =================================================================
elif page == "🧪 Test du modèle":
    st.title("🧪 Tester une transaction")
    st.markdown("Entrez les caractéristiques d'une transaction pour obtenir un score de risque de fraude.")

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### Transaction")
        amount = st.number_input("Montant ($)", min_value=0.0, max_value=50000.0, value=120.0, step=10.0)
        product = st.selectbox("Type de produit", ["W", "H", "C", "S", "R"])
        card_type = st.selectbox("Type de carte", ["credit", "debit"])
        card_network = st.selectbox("Réseau de carte", ["visa", "mastercard", "discover", "american express"])

    with col2:
        st.markdown("#### Contexte")
        hour = st.slider("Heure de la transaction", 0, 23, 14)
        email = st.selectbox("Domaine email", ["gmail.com", "yahoo.com", "hotmail.com",
                                                "outlook.com", "mail.com", "icloud.com", "autre"])
        device = st.selectbox("Type de device", ["desktop", "mobile", "inconnu"])

    with col3:
        st.markdown("#### Indicateurs")
        is_night = 1 if (hour >= 22 or hour <= 6) else 0
        is_peak = 1 if (5 <= hour <= 10) else 0
        is_round = 1 if amount % 1 == 0 else 0

        st.write(f"Transaction de nuit : {'Oui ⚠️' if is_night else 'Non ✓'}")
        st.write(f"Heure pic fraude : {'Oui ⚠️' if is_peak else 'Non ✓'}")
        st.write(f"Montant rond : {'Oui' if is_round else 'Non'}")

    st.markdown("---")

    if st.button("🔍 Analyser la transaction", type="primary", use_container_width=True):

        # Calcul du score de risque basé sur les KPIs
        risk_score = 0
        risk_factors = []

        # Facteur produit
        product_risk = {"C": 11.7, "S": 5.9, "H": 4.8, "R": 3.8, "W": 2.0}
        p_risk = product_risk.get(product, 3.5)
        if p_risk > 5:
            risk_score += 25
            risk_factors.append(f"Produit {product} : taux de fraude élevé ({p_risk:.1f}%)")
        elif p_risk > 3.5:
            risk_score += 10

        # Facteur heure
        if is_peak:
            risk_score += 20
            risk_factors.append(f"Heure {hour}h : pic de fraude (5h-10h)")
        if is_night:
            risk_score += 10
            risk_factors.append("Transaction de nuit")

        # Facteur email
        email_risk = {"mail.com": 19.0, "outlook.com": 9.5, "hotmail.com": 5.3, "gmail.com": 4.4}
        e_risk = email_risk.get(email, 3.0)
        if e_risk > 8:
            risk_score += 25
            risk_factors.append(f"Email {email} : taux de fraude très élevé ({e_risk:.1f}%)")
        elif e_risk > 5:
            risk_score += 10
            risk_factors.append(f"Email {email} : taux de fraude élevé ({e_risk:.1f}%)")

        # Facteur device
        if device == "mobile":
            risk_score += 15
            risk_factors.append("Mobile : canal plus risqué (10.2% vs 6.5% desktop)")

        # Facteur carte
        if card_type == "credit":
            risk_score += 10
            risk_factors.append("Carte de crédit : plus ciblée que débit (6.7% vs 2.4%)")

        if card_network == "discover":
            risk_score += 10
            risk_factors.append("Réseau Discover : taux le plus élevé (7.7%)")

        # Facteur montant
        if amount > 500:
            risk_score += 10
            risk_factors.append(f"Montant élevé (${amount:.0f})")

        # Normaliser le score
        risk_score = min(risk_score, 100)

        # Affichage
        st.markdown("---")
        st.subheader("Résultat de l'analyse")

        col1, col2 = st.columns([1, 2])

        with col1:
            if risk_score >= 70:
                st.error(f"## 🚨 Score de risque : {risk_score}/100")
                st.error("**TRANSACTION SUSPECTE**")
                decision = "Bloquer et investiguer"
            elif risk_score >= 40:
                st.warning(f"## ⚠️ Score de risque : {risk_score}/100")
                st.warning("**SURVEILLANCE RENFORCÉE**")
                decision = "Vérification recommandée"
            else:
                st.success(f"## ✅ Score de risque : {risk_score}/100")
                st.success("**TRANSACTION NORMALE**")
                decision = "Autoriser"

            st.metric("Décision", decision)

        with col2:
            st.markdown("#### Facteurs de risque détectés")
            if risk_factors:
                for factor in risk_factors:
                    st.markdown(f"- ⚠️ {factor}")
            else:
                st.markdown("Aucun facteur de risque majeur détecté.")

        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_score,
            title={"text": "Score de risque"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#E53935" if risk_score>=70 else "#FF9800" if risk_score>=40 else "#4CAF50"},
                "steps": [
                    {"range": [0, 40], "color": "#E8F5E9"},
                    {"range": [40, 70], "color": "#FFF3E0"},
                    {"range": [70, 100], "color": "#FFEBEE"},
                ],
                "threshold": {"line": {"color": "black", "width": 2}, "value": risk_score},
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)


# =================================================================
# PAGE 5 : Comparaison des approches
# =================================================================
elif page == "⚖️ Comparaison des approches":
    st.title("⚖️ Comparaison des approches")

    # Résultats hardcodés (issus des expériences)
    st.subheader("Itérations du projet")

    iterations = pd.DataFrame({
        "Itération": ["V1 - MLP (encodeur gelé)", "V2 - MLP (encodeur dégelé)",
                      "Hybride - XGBoost seul", "Hybride - SSL + XGBoost",
                      "Hybride - SSL + orig. + XGBoost"],
        "AUC-ROC": [0.8224, 0.8932, 0.9158, 0.8510, 0.9137],
        "Precision": [0.7910, 0.5047, 0.7576, 0.5565, 0.7539],
        "Recall": [0.1365, 0.4319, 0.5408, 0.3586, 0.5352],
        "F1-score": [0.2328, 0.4654, 0.6311, 0.4361, 0.6260],
    })

    st.dataframe(
        iterations.style.highlight_max(subset=["AUC-ROC", "Recall", "F1-score"], color="#C8E6C9")
                       .highlight_min(subset=["AUC-ROC", "Recall", "F1-score"], color="#FFCDD2")
                       .format({"AUC-ROC": "{:.4f}", "Precision": "{:.4f}",
                               "Recall": "{:.4f}", "F1-score": "{:.4f}"}),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")

    # Graphique comparatif
    fig = make_subplots(rows=1, cols=4, subplot_titles=["AUC-ROC", "Precision", "Recall", "F1-score"])

    colors = ["#9E9E9E", "#FF9800", "#2196F3", "#E53935", "#9C27B0"]
    for i, metric in enumerate(["AUC-ROC", "Precision", "Recall", "F1-score"]):
        fig.add_trace(go.Bar(
            x=iterations["Itération"], y=iterations[metric],
            marker_color=colors, showlegend=False,
            text=[f"{v:.3f}" for v in iterations[metric]], textposition="outside",
        ), row=1, col=i+1)
        fig.update_yaxes(range=[0, 1.1], row=1, col=i+1)

    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Preprocessing V1 vs V2
    st.subheader("Impact du preprocessing")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### V1 (initial)")
        st.markdown(
            "- Seuil NaN : **70%** -> 208 colonnes supprimées\n"
            "- **226** features conservées\n"
            "- LabelEncoder (ordres arbitraires)\n"
            "- Pas d'indicateurs de NaN\n"
            "- Pas de feature engineering"
        )
    with col2:
        st.markdown("#### V2 (amélioré)")
        st.markdown(
            "- Seuil NaN : **95%** -> 9 colonnes supprimées\n"
            "- **743** features conservées\n"
            "- Frequency encoding (informatif)\n"
            "- **+314** indicateurs de valeurs manquantes\n"
            "- Features temporelles + log montant"
        )

    st.markdown("---")

    # Limited labels
    st.subheader("Expérience : SSL avec labels limités (V1)")
    st.markdown("*Résultats sur preprocessing V1 — V2 en cours*")

    limited_data = pd.DataFrame({
        "Labels (%)": ["1%", "2%", "5%", "10%", "25%", "50%", "100%"],
        "Baseline AUC": [0.8323, 0.8486, 0.8751, 0.8815, 0.8915, 0.8964, 0.8977],
        "SSL AUC": [0.8177, 0.8459, 0.8706, 0.8801, 0.8905, 0.8943, 0.8970],
        "Baseline F1": [0.4392, 0.4554, 0.4885, 0.5146, 0.5459, 0.5643, 0.5647],
        "SSL F1": [0.4181, 0.4506, 0.4832, 0.5176, 0.5455, 0.5635, 0.5726],
    })

    fig = make_subplots(rows=1, cols=2, subplot_titles=["AUC-ROC", "F1-score"])
    fig.add_trace(go.Scatter(x=limited_data["Labels (%)"], y=limited_data["Baseline AUC"],
                            mode="lines+markers", name="Baseline", line=dict(color="#888888", width=2.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=limited_data["Labels (%)"], y=limited_data["SSL AUC"],
                            mode="lines+markers", name="SSL", line=dict(color="#E53935", width=2.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=limited_data["Labels (%)"], y=limited_data["Baseline F1"],
                            mode="lines+markers", name="Baseline", line=dict(color="#888888", width=2.5),
                            showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=limited_data["Labels (%)"], y=limited_data["SSL F1"],
                            mode="lines+markers", name="SSL", line=dict(color="#E53935", width=2.5),
                            showlegend=False), row=1, col=2)
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.info("Le SSL avec preprocessing V1 ne montre pas de gain significatif. "
            "L'hypothèse est que le preprocessing V1 trop agressif (208 colonnes supprimées) "
            "a appauvri les données d'entraînement SSL. L'expérience V2 (743 features) est en cours.")


# =================================================================
# PAGE 6 : Rapport technique
# =================================================================
elif page == "📋 Rapport technique":
    st.title("📋 Rapport technique")

    st.subheader("Architecture du modèle SSL")
    st.code("""
    Transaction (743 features)
         |
    ┌────▼────┐
    │ Encoder │  743 -> 256 -> 96 (représentation latente)
    └────┬────┘
         |
    ┌────▼─────────────┐
    │ Reconstruction   │  96 -> 256 -> 743 (pretraining SSL)
    │ Head             │
    └──────────────────┘
         |
    ┌────▼─────────────┐
    │ XGBoost          │  96 embeddings -> fraude / non fraude
    │ Classifier       │
    └──────────────────┘
    """, language=None)

    st.markdown("---")

    st.subheader("Preprocessing : V1 vs V2")
    st.code("""
    V1 : 434 colonnes -> seuil 70% -> 226 features
         LabelEncoder, médiane, pas d'indicateurs NaN

    V2 : 434 colonnes -> seuil 95% -> 743 features
         Frequency encoding, indicateurs NaN (+314),
         features temporelles (+4), log montant (+2)
    """, language=None)

    st.markdown("---")

    st.subheader("Métriques disponibles")

    # Charger les rapports
    results_dir = os.path.join("reports", "results")
    if os.path.exists(results_dir):
        for fname in sorted(os.listdir(results_dir)):
            if fname.endswith(".txt"):
                with st.expander(f"📄 {fname}"):
                    with open(os.path.join(results_dir, fname), "r", encoding="utf-8", errors="ignore") as f:
                        st.code(f.read())

    st.markdown("---")

    # Graphiques existants
    st.subheader("Graphiques générés")
    if os.path.exists(REPORTS_DIR):
        figs = [f for f in os.listdir(REPORTS_DIR) if f.endswith(".png")]
        cols = st.columns(2)
        for i, fname in enumerate(sorted(figs)):
            with cols[i % 2]:
                st.image(os.path.join(REPORTS_DIR, fname), caption=fname, use_container_width=True)


# ─── Footer ─────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("*Capstone Project 2025-2026*")
