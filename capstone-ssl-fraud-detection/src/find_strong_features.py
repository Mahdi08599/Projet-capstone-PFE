"""
Recherche de features fortes pour améliorer le modèle
"""
import pandas as pd
import numpy as np

df = pd.read_csv("../data/raw/train_transaction.csv")
id_df = pd.read_csv("../data/raw/train_identity.csv")
df = df.merge(id_df, on="TransactionID", how="left")

print("=== RECHERCHE DE FEATURES FORTES ===\n")

# 1. Frequence par carte
freq = df.groupby("card1")["TransactionID"].transform("count")
df["card1_freq"] = freq
high_freq = df[df["card1_freq"] > df["card1_freq"].quantile(0.95)]
low_freq = df[df["card1_freq"] <= df["card1_freq"].quantile(0.95)]
r1 = high_freq["isFraud"].mean() / max(low_freq["isFraud"].mean(), 0.001)
print(f"1. Frequence carte (card1):")
print(f"   Haute freq  : {high_freq['isFraud'].mean()*100:.2f}%")
print(f"   Basse freq  : {low_freq['isFraud'].mean()*100:.2f}%")
print(f"   Ratio : {r1:.1f}x")

# 2. Carte utilisee a plusieurs adresses
combo = df.groupby("card1")["addr1"].transform("nunique")
df["card1_n_addr"] = combo
multi = df[df["card1_n_addr"] > 1]
single = df[df["card1_n_addr"] == 1]
r2 = multi["isFraud"].mean() / max(single["isFraud"].mean(), 0.001)
print(f"\n2. Carte utilisee a plusieurs adresses:")
print(f"   Multi-adresse  : {multi['isFraud'].mean()*100:.2f}%")
print(f"   Mono-adresse   : {single['isFraud'].mean()*100:.2f}%")
print(f"   Ratio : {r2:.1f}x")

# 3. Email mismatch
df["email_mismatch"] = (df["P_emaildomain"] != df["R_emaildomain"]).astype(int)
match = df[df["email_mismatch"] == 0]
mismatch = df[df["email_mismatch"] == 1]
r3 = mismatch["isFraud"].mean() / max(match["isFraud"].mean(), 0.001)
print(f"\n3. Email mismatch (acheteur != destinataire):")
print(f"   Mismatch : {mismatch['isFraud'].mean()*100:.2f}%")
print(f"   Match    : {match['isFraud'].mean()*100:.2f}%")
print(f"   Ratio : {r3:.1f}x")

# 4. D1 extreme
if "D1" in df.columns:
    q95 = df["D1"].quantile(0.95)
    d1_high = df[df["D1"] > q95]
    d1_low = df[df["D1"] <= q95]
    r4 = d1_high["isFraud"].mean() / max(d1_low["isFraud"].mean(), 0.001)
    print(f"\n4. D1 extreme (>95 percentile):")
    print(f"   D1 extreme : {d1_high['isFraud'].mean()*100:.2f}%")
    print(f"   D1 normal  : {d1_low['isFraud'].mean()*100:.2f}%")
    print(f"   Ratio : {r4:.1f}x")

# 5. Nombre de NaN par transaction
n_nan = df.isnull().sum(axis=1)
df["n_missing"] = n_nan
med = df["n_missing"].median()
high_m = df[df["n_missing"] > med]
low_m = df[df["n_missing"] <= med]
r5 = high_m["isFraud"].mean() / max(low_m["isFraud"].mean(), 0.001)
print(f"\n5. Transactions avec beaucoup de NaN:")
print(f"   Beaucoup NaN : {high_m['isFraud'].mean()*100:.2f}%")
print(f"   Peu NaN      : {low_m['isFraud'].mean()*100:.2f}%")
print(f"   Ratio : {r5:.1f}x")

# 6. Montant cents (fraudes souvent en montants ronds)
df["cents"] = df["TransactionAmt"] % 1
round_amt = df[df["cents"] == 0]
decimal_amt = df[df["cents"] != 0]
r6 = round_amt["isFraud"].mean() / max(decimal_amt["isFraud"].mean(), 0.001)
print(f"\n6. Montant rond vs avec centimes:")
print(f"   Montant rond    : {round_amt['isFraud'].mean()*100:.2f}%")
print(f"   Avec centimes   : {decimal_amt['isFraud'].mean()*100:.2f}%")
print(f"   Ratio : {r6:.1f}x")

# 7. Heure peak
df["hour"] = (df["TransactionDT"] / 3600 % 24).astype(int)
peak = df[(df["hour"] >= 5) & (df["hour"] <= 10)]
offpeak = df[(df["hour"] < 5) | (df["hour"] > 10)]
r7 = peak["isFraud"].mean() / max(offpeak["isFraud"].mean(), 0.001)
print(f"\n7. Heure de pointe fraude (5h-10h):")
print(f"   Peak     : {peak['isFraud'].mean()*100:.2f}%")
print(f"   Off-peak : {offpeak['isFraud'].mean()*100:.2f}%")
print(f"   Ratio : {r7:.1f}x")

# Resume
print(f"\n{'='*50}")
print(f"RESUME DES SIGNAUX :")
print(f"{'='*50}")
signals = [
    ("Frequence carte", r1),
    ("Multi-adresse", r2),
    ("Email mismatch", r3),
    ("D1 extreme", r4 if "D1" in df.columns else 0),
    ("Beaucoup NaN", r5),
    ("Montant rond", r6),
    ("Heure peak", r7),
]
signals.sort(key=lambda x: x[1], reverse=True)
for name, ratio in signals:
    stars = "***" if ratio >= 3 else "**" if ratio >= 2 else "*"
    print(f"  {name:<25} {ratio:.1f}x {stars}")
