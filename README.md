# Prêt à dépenser – Scoring crédit (MLOps)

Mise en production d'un modèle de scoring crédit (LightGBM) qui prédit la probabilité
de défaut d'un client et décide si le crédit est accordé ou refusé.

## Le modèle
- LightGBM optimisé avec Optuna, suivi et versionné avec MLflow
  (`credit_scoring_lgbm` v2, alias `champion`)
- 796 features issues des 8 tables Home Credit (feature engineering : `notebooks/`)
- Seuil de décision métier : **0,08** (coût = 10 × FN + 1 × FP)
- Hold-out : AUC 0,786, recall 71 %, coût métier 0,491

## Structure
| Dossier / fichier | Contenu |
|---|---|
| `notebooks/` | Exploration, feature engineering, modélisation (partie 1) |
| `model/` | Modèle champion exporté depuis le Model Registry MLflow |
| `data/sample/` | 1000 clients du jeu de test Kaggle (pseudonymisés), aux features déjà préparées |
| `scripts/` | Export du modèle et création de l'échantillon |
| `src/` | Script d'inférence |
| `requirements.txt` | Dépendances minimales pour l'inférence (API, Docker) |
| `requirements-notebooks.txt` | Dépendances complètes pour rejouer les notebooks |

## Installation
```bash
pip install -r requirements.txt
python src/predict.py 100001
```

Python utilisé : 3.11.9

## Données
Les données brutes (Kaggle Home Credit) et les fichiers complets ne sont pas versionnés.
Seul un échantillon de 1000 clients est inclus, pour faire tourner l'API et les tests.
Préparation des données adaptée du kernel Kaggle « LightGBM with Simple Features » (jsaguiar).