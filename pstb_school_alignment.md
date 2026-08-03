# Alignement du memoire avec PST&B

## Source officielle

Site officiel PST&B : https://www.pstb.fr/

Elements retenus :

- PST&B signifie Paris School of Technology & Business.
- L'ecole positionne ses formations autour de la combinaison entre technologie et business.
- Le Mastère Data Science in Business a pour objectif de former des experts de l'exploitation, de l'analyse et de l'interpretation des donnees.
- Le programme met en avant l'utilisation d'outils statistiques et mathematiques, notamment des algorithmes, afin d'orienter la strategie des entreprises.

## Consequence pour notre memoire

Le memoire ne doit pas etre redige comme un simple projet Kaggle ou un simple benchmark technique. Il doit etre presente comme un projet Data Science in Business, c'est-a-dire :

- partir d'un probleme business concret : la fraude dans la banque digitale ;
- construire une chaine data complete : collecte, ETL leger, preprocessing, modelisation, evaluation ;
- justifier les choix de metriques par le contexte metier ;
- traduire les performances ML en KPIs exploitables : fraudes detectees, fausses alertes, cout d'investigation, benefice net ;
- proposer une interface de decision : dashboard Streamlit et assistant metier ;
- discuter les limites et les perspectives dans une logique operationnelle.

## Angle de redaction recommande

La ligne narrative du memoire doit etre :

> Ce projet illustre la mise en oeuvre d'une demarche Data Science in Business appliquee a la banque digitale. Il ne se limite pas a entrainer un modele de machine learning : il transforme un probleme de risque financier en systeme d'aide a la decision, combinant donnees, modelisation, indicateurs metier et visualisation interactive.

## Formulation prete pour l'introduction

Dans le cadre du Mastère Data Science in Business de PST&B, ce projet s'inscrit dans une logique de croisement entre expertise technologique et finalite business. L'objectif n'est pas uniquement de developper un modele predictif, mais de concevoir une solution exploitable permettant a une banque digitale d'ameliorer la detection de fraude, de reduire les pertes financieres et d'appuyer la prise de decision par des indicateurs metier comprehensibles.

Le choix d'un cas d'usage bancaire se justifie par la place centrale des donnees transactionnelles dans les services financiers digitaux. La fraude constitue un risque operationnel et financier majeur, qui necessite des methodes capables d'analyser de grands volumes de transactions, de gerer le desequilibre naturel entre operations legitimes et frauduleuses, et de transformer les scores de risque en decisions actionnables.

## Implication sur la structure du memoire

Pour etre coherent avec PST&B, chaque chapitre doit garder un lien explicite entre technique et business :

- Introduction : contexte banque digitale, risque financier, problematique data/business.
- Etat de l'art : fraude transactionnelle, donnees desequilibrees, SSL, XGBoost, evaluation metier.
- Donnees : justification du dataset, lien avec transactions digitales, variables metier.
- Methodologie : ETL, preprocessing, modeles, seuil de decision.
- Resultats : metriques ML + interpretation business.
- Dashboard : outil de restitution, simulation et aide a la decision.
- Conclusion : apports, limites, deploiement, perspectives metier.

## Phrase de defense pour la soutenance

> Notre projet repond a l'esprit Data Science in Business de PST&B : nous avons transforme un probleme technique de classification desequilibree en un dispositif de decision pour banque digitale, avec un modele performant, des KPIs financiers et un dashboard deploye permettant de rendre les resultats exploitables par des profils non techniques.
