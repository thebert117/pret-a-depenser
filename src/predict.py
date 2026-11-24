"""Logique d'inférence : chargement du modèle, validation et préparation des entrées, prédiction.

Usage en ligne de commande : python src/predict.py 100001
"""
import sys
from pathlib import Path

import mlflow.pyfunc
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "model"
DATA_PATH = ROOT / "data" / "sample" / "clients_sample.parquet"

# Champs obligatoires : toujours renseignés dans une demande de crédit
OBLIGATOIRES = ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "DAYS_BIRTH"]

# Plages de valeurs plausibles (DAYS_BIRTH en jours négatifs : entre 18 et 100 ans)
BORNES = {
    "EXT_SOURCE_1": (0, 1),
    "EXT_SOURCE_2": (0, 1),
    "EXT_SOURCE_3": (0, 1),
    "AMT_INCOME_TOTAL": (1, 1e9),
    "AMT_CREDIT": (1, 1e8),
    "AMT_ANNUITY": (1, 1e7),
    "DAYS_BIRTH": (-36500, -6570),
    "CNT_CHILDREN": (0, 20),
}


def charger_modele(chemin=MODEL_PATH):
    """Charge le modèle champion exporté (wrapper ScoringCredit)."""
    return mlflow.pyfunc.load_model(str(chemin))


def charger_clients(chemin=DATA_PATH):
    """Charge les features préparées des clients, indexées par SK_ID_CURR."""
    return pd.read_parquet(chemin).set_index("SK_ID_CURR")


def types_attendus(modele):
    """Renvoie {colonne: dtype} lu dans la signature MLflow du modèle (ordre conservé)."""
    return {c.name: c.type.to_pandas() for c in modele.metadata.get_input_schema().inputs}


def est_colonne_discrete(dtype):
    """Colonnes entières ou booléennes (one-hot, FLAG_) : pas de NaN autorisé."""
    return not np.issubdtype(dtype, np.floating)


def valider_features(features, types):
    """Renvoie la liste des erreurs de validation (liste vide si tout est correct)."""
    erreurs = []

    inconnues = sorted(set(features) - set(types))
    if inconnues:
        erreurs.append(f"Features inconnues du modèle : {inconnues[:10]}")

    for col in OBLIGATOIRES:
        if col in types and features.get(col) is None:
            erreurs.append(f"{col} est obligatoire")

    for col, (mini, maxi) in BORNES.items():
        valeur = features.get(col)
        if col in types and valeur is not None and not mini <= valeur <= maxi:
            erreurs.append(f"{col} = {valeur} hors de la plage autorisée [{mini}, {maxi}]")

    for col, valeur in features.items():
        if col in types and valeur is not None and est_colonne_discrete(types[col]):
            if float(valeur) != int(valeur):
                erreurs.append(f"{col} doit être un entier")

    return erreurs


def preparer_features(features, types):
    """Construit la ligne complète attendue par le modèle à partir d'un dict partiel.

    Features absentes : NaN pour les colonnes décimales (LightGBM gère nativement),
    0 pour les colonnes discrètes (catégorie absente / flag non coché).
    """
    ligne = {}
    for col, dtype in types.items():
        valeur = features.get(col)
        if valeur is None:
            valeur = 0 if est_colonne_discrete(dtype) else np.nan
        ligne[col] = valeur
    return pd.DataFrame([ligne]).astype(types)


def predire(modele, X):
    """Renvoie la probabilité de défaut et la décision pour une ligne de features."""
    resultat = modele.predict(X).iloc[0]
    return {
        "probabilite_defaut": float(resultat["probabilite_defaut"]),
        "decision": str(resultat["decision"]),
    }


def predire_client(modele, clients, types, sk_id_curr):
    """Prédiction pour un client connu, à partir de son SK_ID_CURR."""
    if sk_id_curr not in clients.index:
        raise KeyError(f"Client {sk_id_curr} introuvable")
    X = clients.loc[[sk_id_curr], list(types)]
    return predire(modele, X)


if __name__ == "__main__":
    sk_id = int(sys.argv[1]) if len(sys.argv) > 1 else 100001
    modele = charger_modele()
    print(predire_client(modele, charger_clients(), types_attendus(modele), sk_id))