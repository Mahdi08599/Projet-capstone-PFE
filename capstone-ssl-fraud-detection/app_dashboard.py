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


st.sidebar.title("🛡️ Navigation")
page = st.sidebar.radio("", [
    "🏠 Vue d'ensemble", "🔍 Exploration des données", "📈 KPIs Business",
    "🧪 Test du modèle", "📊 Performances", "🔬 Itérations du projet",
])

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎯 Modèle final")
st.sidebar.metric("AUC-ROC", f"{FINAL_METRICS['auc']:.4f}")
st.sidebar.metric("F1-score", f"{FINAL_METRICS['f1']:.4f}")
st.sidebar.metric("Recall", f"{FINAL_METRICS['recall']:.1%}")
st.sidebar.metric("Precision", f"{FINAL_METRICS['precision']:.1%}")
st.sidebar.markdown("---")
st.sidebar.markdown("**Capstone PFE — Master Data/AI**")
st.sidebar.caption("EL HAMDAOUI Mohamed · BEN ARAFI Mahdi")
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
        st.info("**Comment le self-supervised learning peut-il améliorer la détection "
                "de fraude dans la banque digitale lorsque les données labellisées sont limitées ?**")
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
        st.subheader("Distribution horaire des fraudes")
        df["hour"] = (df["TransactionDT"]/3600%24).astype(int)
        hourly = df.groupby("hour").agg(total=("isFraud","count"), frauds=("isFraud","sum")).reset_index()
        hourly["taux"] = hourly["frauds"]/hourly["total"]*100
        fig = px.bar(hourly, x="hour", y="taux", color="taux",
                    color_continuous_scale=["#22c55e","#eab308","#dc2626"])
        fig.update_layout(height=350, showlegend=False, xaxis=dict(dtick=1))
        st.plotly_chart(fig, use_container_width=True)
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
        st.markdown("#### 🚩 Signaux")
        is_night = 1 if (hour>=22 or hour<=6) else 0
        is_peak = 1 if (5<=hour<=10) else 0
        st.write(f"Nuit : {'⚠️ Oui' if is_night else '✅ Non'}")
        st.write(f"Pic fraude (5-10h) : {'⚠️ Oui' if is_peak else '✅ Non'}")
        st.write(f"Montant rond : {'Oui' if amount%1==0 else 'Non'}")
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
    st.json(BEST_PARAMS)


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