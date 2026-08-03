# Synthese technique du projet capstone

Ce document consolide les elements utiles issus des discussions de travail. Il sert de source interne pour le memoire et la soutenance.

## Positionnement du projet

Le projet vise la detection de fraude dans un contexte de banque digitale. La demarche retenue est hybride : exploration du self-supervised learning pour apprendre des representations transactionnelles, puis choix operationnel d'un modele XGBoost optimise avec seuil metier et dashboard d'aide a la decision.

Problematique retenue :

> Comment concevoir un systeme hybride de detection de fraude bancaire digitale, combinant self-supervised learning, apprentissage supervise et optimisation metier, afin d'ameliorer la performance, l'explicabilite et l'exploitabilite des decisions dans un contexte de donnees fortement desequilibrees ?

## Dataset principal

Dataset : IEEE-CIS Fraud Detection.

Fichiers principaux :

- `train_transaction.csv` : 590 540 transactions, 394 colonnes, contient `isFraud`.
- `train_identity.csv` : 144 233 lignes, 41 colonnes, jointure sur `TransactionID`.
- `test_transaction.csv`, `test_identity.csv`, `sample_submission.csv`.

Caracteristiques :

- Taux de fraude : 3.50%, soit 20 663 fraudes sur 590 540 transactions.
- Fort desequilibre des classes : environ 1 fraude pour 27 transactions legitimes.
- Donnees tabulaires mixtes : montant, produit, carte, adresse, email, variables anonymisees Vesta, device.
- Donnees identity disponibles pour environ 24% des transactions.

## ETL leger et preprocessing

Un vrai data warehouse n'a pas ete retenu, car le projet repose sur un dataset fixe Kaggle. La demarche pertinente est un ETL leger :

- Extract : chargement des CSV et jointure transaction + identity.
- Transform : nettoyage, encodage, feature engineering, gestion des valeurs manquantes.
- Load : sauvegarde des donnees pretraitees en Parquet pour les modeles.

### Preprocessing V1

Objectif : obtenir rapidement une premiere base exploitable.

- Suppression des colonnes avec plus de 70% de valeurs manquantes.
- Imputation mediane pour les numeriques.
- Imputation `UNKNOWN` pour les categories.
- Label encoding des variables categorielles.
- StandardScaler.
- Split train/validation 80/20 stratifie.

Limite identifiee : suppression trop agressive de colonnes potentiellement informatives et perte du signal porte par les valeurs manquantes.

### Preprocessing V2

Objectif : corriger l'appauvrissement de V1.

- Seuil de suppression releve a 95% de valeurs manquantes.
- Conservation de beaucoup plus de variables Vesta.
- Ajout d'indicateurs de valeurs manquantes.
- Frequency encoding pour certaines categories.
- Features temporelles : heure, nuit, pic de fraude.
- Features metier : montant log-transforme, montant rond, etc.

Resultat : passage a environ 743 features apres transformation.

Justification du seuil 95% : seuil pragmatique permettant de conserver les colonnes avec suffisamment d'observations exploitables, tout en evitant les colonnes quasi vides.

## Iterations modeles

### SSL V1

Architecture :

- Autoencoder MLP.
- Masquage aleatoire d'environ 15% des features.
- Objectif : reconstruire les variables masquees sans utiliser `isFraud`.

Resultat : l'encodeur apprend une representation, mais la classification fraude reste faible.

Constat : le SSL seul ne suffit pas pour la decision fraude sur ces donnees tabulaires.

### Fine-tuning supervise

Premiere correction :

- Degel de l'encodeur.
- Loss ponderee avec `pos_weight` pour compenser le desequilibre.
- Optimisation du seuil de decision.

Gain : amelioration nette du recall par rapport a la version initiale, mais performance encore insuffisante pour un systeme metier robuste.

### Comparaison SSL vs XGBoost

Experience realisee :

- XGBoost sur features originales.
- XGBoost sur embeddings SSL.
- XGBoost sur features originales + embeddings SSL.
- Experiences en labels limites : 1%, 2%, 5%, 10%, etc.

Conclusion scientifique :

- Le SSL n'apporte pas de gain significatif sur ce dataset tabulaire.
- Les features tabulaires sont deja fortement informatives pour XGBoost.
- Ce resultat negatif est defendable : le SSL est plus mature en vision/NLP qu'en donnees tabulaires structurees.

Position a tenir :

Le SSL reste valorise comme approche exploratoire initiale et comme composante scientifique de la demarche. Le systeme final est hybride dans la demarche : SSL explore et documente, XGBoost optimise assure la decision operationnelle.

## Modele final retenu

Modele final : XGBoost optimise sur les donnees V2.

Resultats finaux a utiliser dans le memoire et le dashboard :

- AUC-ROC : 0.9718.
- Average Precision / PR-AUC : 0.8608.
- F1-score : 0.8210.
- Precision : 0.8797.
- Recall : 0.7697.
- Seuil de decision final : 0.56.

Hyperparametres finaux :

- `colsample_bytree` : 0.7364265404201034.
- `gamma` : 0.056736760620294535.
- `learning_rate` : 0.14870404274178442.
- `max_depth` : 9.
- `min_child_weight` : 3.
- `n_estimators` : 562.
- `reg_alpha` : 0.659984046034179.
- `reg_lambda` : 2.1344444004024314.
- `subsample` : 0.822080324639785.

Important : les anciens seuils 0.70, 0.74, 0.78, 0.83 et les resultats associes sont des analyses intermediaires. Le seuil final retenu pour coherence projet est 0.56.

## KPIs metier principaux

KPIs dataset :

- Transactions : 590 540.
- Nombre de fraudes : 20 663.
- Taux de fraude : 3.50%.
- Volume frauduleux : 3 083 845 USD.
- Montant moyen fraude : 149.24 USD.

KPIs produits :

- Produit C : 11.69% de fraude, 8 008 fraudes sur 68 519 transactions.
- Produit W : 2.04% de fraude, 8 969 fraudes sur 439 670 transactions.
- Lecture importante : W a le taux de fraude le plus faible, mais son volume massif explique son grand nombre de fraudes absolues.

KPIs modeles projetes sur dataset complet :

- Fraudes detectees estimees : 15 904 sur 20 663.
- Fausses alertes estimees : 2 174.
- Montant sauve estime : 2 373 635 USD.
- Cout d'investigation estime : 271 170 USD.
- Benefice net estime : 2 102 465 USD.

## Analyse des fraudes ratees

Point important pour le memoire :

Les fraudes ratees ne sont pas seulement des erreurs aleatoires. Elles ressemblent davantage aux transactions legitimes :

- Forte presence du produit W.
- Usage plus frequent de cartes debit.
- Moins de presence dans les heures de pic.
- Domaines email courants.
- Absence ou faiblesse de signaux device.

Interpretation :

Le modele detecte efficacement les comportements frauduleux atypiques. Les fraudes restantes sont plus difficiles car elles imitent les comportements legitimes. Cela justifie des perspectives comme l'ajout de donnees comportementales, historiques client, IP, geolocalisation ou verification d'identite.

## Dashboard Streamlit

Le dashboard doit rester synchronise avec le modele final :

- Vue d'ensemble : metriques finales et problematique.
- Exploration : dataset et desequilibre.
- KPIs Business : graphiques produits, exposition, impact financier.
- Test du modele : simulation d'une transaction.
- Performances : resultats finaux et hyperparametres vulgarises.
- Assistant metier : chatbot local, sans invention, base sur les rapports et les metriques validees.

URL de deploiement :

`https://fraud-detection-bank-ssl.streamlit.app`

## Regle de redaction

Ne pas presenter les resultats intermediaires comme resultats finaux. Les utiliser pour raconter la progression :

1. Baseline et SSL initial.
2. Identification des limites.
3. Preprocessing V2.
4. Comparaison SSL / XGBoost.
5. Optimisation XGBoost.
6. Choix du seuil metier.
7. Traduction en KPIs business et dashboard.

Message cle pour la soutenance :

Le projet ne se limite pas a entrainer un modele. Il construit une chaine complete de decision pour banque digitale : donnees, experimentation scientifique, modele performant, seuil metier, impact financier, explicabilite et interface de demonstration.
