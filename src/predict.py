"""Script d'inférence : score un client à partir de son SK_ID_CURR.

Usage : python src/predict.py 100001
"""
import sys
from pathlib import Path

import mlflow.pyfunc
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "model"
DATA_PATH = ROOT / "data" / "sample" / "clients_sample.parquet"


def charger_modele():
    """Charge le modèle champion exporté (wrapper ScoringCredit)."""
    return mlflow.pyfunc.load_model(str(MODEL_PATH))


def charger_clients():
    """Charge les features préparées des clients, indexées par SK_ID_CURR."""
    return pd.read_parquet(DATA_PATH).set_index("SK_ID_CURR")


def predire(modele, clients, sk_id_curr):
    """Renvoie la probabilité de défaut et la décision pour un client."""
    if sk_id_curr not in clients.index:
        raise KeyError(f"Client {sk_id_curr} introuvable")
    colonnes = modele.metadata.get_input_schema().input_names()
    X = clients.loc[[sk_id_curr], colonnes]
    return modele.predict(X).iloc[0].to_dict()


if __name__ == "__main__":
    sk_id = int(sys.argv[1]) if len(sys.argv) > 1 else 100001
    print(predire(charger_modele(), charger_clients(), sk_id))