# Recherche web et justification du choix du dataset

## Objectif de la recherche

Cette note justifie le choix du dataset IEEE-CIS Fraud Detection comme base principale du projet capstone. Elle peut etre reprise dans le memoire, principalement dans la partie "Choix du jeu de donnees", "Cadre experimental" ou "Methodologie".

## Dataset retenu

Le dataset principal retenu est **IEEE-CIS Fraud Detection**, publie sur Kaggle dans le cadre d'une competition associee a l'IEEE Computational Intelligence Society et a Vesta, acteur specialise dans la prevention de fraude sur les paiements digitaux.

Source principale :

- Kaggle, IEEE-CIS Fraud Detection : https://www.kaggle.com/c/ieee-fraud-detection

## Pourquoi ce dataset est pertinent pour une banque digitale

### 1. Il correspond a un cas de fraude transactionnelle digitale

Le projet vise la detection de fraude dans un environnement de banque digitale. Le dataset IEEE-CIS contient des transactions numeriques avec des variables liees au paiement, au produit, a la carte, aux emails, au temps et a l'identite/device. Cette structure correspond a une logique de monitoring transactionnel : chaque transaction doit etre analysee afin d'estimer son niveau de risque.

Cette approche est coherente avec la litterature sur la fraude bancaire et la fraude carte. Lucas et Jurgovsky expliquent qu'un probleme classique de detection de fraude transactionnelle repose sur des attributs de transaction, de porteur/carte et de marchand, ainsi que sur un label binaire fraude/non-fraude. Ils soulignent aussi que ces donnees sont sensibles et rarement accessibles publiquement, ce qui rend les datasets publics exploitables particulierement importants pour la recherche.

Source :

- Lucas, Y. et Jurgovsky, J. (2020), *Credit card fraud detection using machine learning: A survey* : https://arxiv.org/abs/2010.06479

### 2. Il presente un desequilibre fort, realiste pour la fraude

Dans notre projet, le dataset contient 590 540 transactions dont 20 663 fraudes, soit un taux de fraude de 3.50%. Ce desequilibre est central dans la fraude bancaire : la majorite des transactions sont legitimes, et la classe frauduleuse est rare mais critique.

Ce point justifie :

- l'utilisation de metriques adaptees comme PR-AUC, recall, precision et F1-score ;
- l'optimisation du seuil de decision ;
- la ponderation du desequilibre via `scale_pos_weight` dans XGBoost ;
- une lecture metier des faux positifs et faux negatifs.

Saito et Rehmsmeier montrent que les courbes Precision-Recall sont particulierement utiles dans les contextes fortement desequilibres, car elles mesurent directement la qualite des predictions positives, contrairement a une lecture ROC qui peut paraitre trop optimiste.

Source :

- Saito, T. et Rehmsmeier, M. (2015), *The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets*, PLOS ONE : https://doi.org/10.1371/journal.pone.0118432

### 3. Il permet une lecture metier et pas seulement technique

Le dataset ne contient pas uniquement des variables anonymisees. Il contient aussi des variables interpretables ou semi-interpretables :

- `TransactionAmt` : montant de la transaction ;
- `ProductCD` : produit ou canal transactionnel ;
- `card4`, `card6` : reseau et type de carte ;
- `P_emaildomain`, `R_emaildomain` : domaines email ;
- `TransactionDT` : temps relatif ;
- variables d'identite/device issues de `train_identity`.

Ces variables rendent possible la construction de KPIs metier :

- taux de fraude par produit ;
- volume frauduleux ;
- fraude par type de carte ;
- fraude par heure ;
- impact financier ;
- cout d'investigation et benefice net estime.

Ce point est important pour un PFE Data Science in Business : le dataset permet de passer d'un modele ML a une logique de decision bancaire exploitable.

### 4. Il est reconnu comme benchmark de recherche

IEEE-CIS est utilise dans des travaux recents comme benchmark pour la fraude transactionnelle et les donnees desequilibrees. Par exemple, Han et Wu l'utilisent pour evaluer des approches de fusion de modeles sur un probleme de fraude carte fortement desequilibre. Leur article rappelle que la fraude carte est rare, couteuse et mal distribuee, et que les modeles de gradient boosting restent tres performants sur les donnees transactionnelles structurees.

Source :

- Han, X. et Wu, C. (2026), *Validation-Stage Combinatorial Fusion Analysis for Imbalanced Credit-Card Fraud Detection* : https://arxiv.org/abs/2606.10393

### 5. Il est compatible avec notre demarche hybride SSL + XGBoost

Le dataset contient un grand volume de transactions et beaucoup de variables. Cela permet de tester deux logiques :

- une approche self-supervised learning, qui apprend des representations a partir des transactions sans utiliser le label fraude ;
- une approche supervisee XGBoost, plus adaptee aux donnees tabulaires structurees et a l'optimisation metier.

La litterature sur les plateformes de paiement digital insiste sur le fait que les labels de fraude sont couteux a obtenir, car ils dependent souvent d'investigations humaines. FraudJudger, par exemple, rappelle que les methodes supervisees ont besoin de grandes bases labellisees, difficiles a constituer dans la realite, et explore l'apprentissage de representations a partir de donnees non labellisees.

Source :

- Deng, R. et Ruan, N. (2019), *FraudJudger: Real-World Data Oriented Fraud Detection on Digital Payment Platforms* : https://arxiv.org/abs/1909.02398

## Pourquoi nous n'avons pas retenu un autre dataset comme base principale

### European Credit Card Fraud

Le dataset European Credit Card Fraud est tres connu, mais il contient surtout des variables PCA anonymisees (`V1` a `V28`). Il est utile pour comparer des algorithmes, mais moins adapte a notre objectif business, car il limite fortement l'interpretation metier : produit, canal, email, device et typologie transactionnelle ne sont pas directement disponibles.

### PaySim

PaySim est interessant pour le mobile money et la banque digitale, mais il s'agit d'un dataset simule. Il est pertinent pour une extension ou une validation secondaire, mais moins fort que IEEE-CIS comme dataset principal si l'objectif est de travailler sur des transactions reelles et des variables riches.

### Donnees scrapees ou generees

Les transactions bancaires reelles ne sont pas librement disponibles sur le web pour des raisons de confidentialite. Generer ou scraper des donnees artificielles affaiblirait la credibilite scientifique du memoire, sauf si l'objectif etait explicitement une simulation. Pour ce projet, il est plus solide de travailler sur un benchmark public reconnu.

## Formulation prete pour le memoire

Le choix du dataset IEEE-CIS Fraud Detection se justifie par son adequation avec la problematique de detection de fraude en banque digitale. Il s'agit d'un jeu de donnees transactionnel de grande taille, contenant 590 540 transactions et un label binaire `isFraud`, avec un taux de fraude de 3.50%. Cette rarete de la fraude reproduit une contrainte centrale des systemes bancaires reels : la majorite des operations sont legitimes, tandis que les transactions frauduleuses sont peu nombreuses mais a fort impact financier.

Contrairement a certains datasets de fraude carte fortement anonymises, IEEE-CIS offre a la fois des variables techniques et des variables interpretables, telles que le montant, le produit, le type de carte, le domaine email, le temps transactionnel et certaines informations d'identite/device. Cette richesse permet non seulement d'entrainer un modele de detection, mais aussi de produire une analyse metier : taux de fraude par produit, volume frauduleux, risque par canal, impact financier et cout d'investigation.

Le dataset est egalement coherent avec la demarche scientifique du projet. Sa taille et son desequilibre permettent de tester une approche self-supervised learning pour apprendre des representations sans labels, puis de comparer cette approche a un modele supervise XGBoost optimise. Les resultats obtenus montrent finalement que XGBoost, combine a un preprocessing adapte et a une strategie de seuil metier, fournit la solution la plus exploitable dans ce contexte tabulaire. Le dataset permet donc de relier experimentation technique, contraintes de donnees desequilibrees et decision business dans un cadre proche des besoins d'une banque digitale.

## Sources retenues

- Kaggle, IEEE-CIS Fraud Detection : https://www.kaggle.com/c/ieee-fraud-detection
- Lucas, Y. et Jurgovsky, J. (2020), *Credit card fraud detection using machine learning: A survey* : https://arxiv.org/abs/2010.06479
- Saito, T. et Rehmsmeier, M. (2015), *The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets* : https://doi.org/10.1371/journal.pone.0118432
- Deng, R. et Ruan, N. (2019), *FraudJudger: Real-World Data Oriented Fraud Detection on Digital Payment Platforms* : https://arxiv.org/abs/1909.02398
- Han, X. et Wu, C. (2026), *Validation-Stage Combinatorial Fusion Analysis for Imbalanced Credit-Card Fraud Detection* : https://arxiv.org/abs/2606.10393
