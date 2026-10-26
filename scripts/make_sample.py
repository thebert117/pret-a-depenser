"""Crée un échantillon de 1000 clients de test_final.parquet.

Seules les colonnes attendues par le modèle sont conservées (+ SK_ID_CURR).
"""
from pathlib import Path

import mlflow.pyfunc
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "processed" / "test_final.parquet"
SORTIE = ROOT / "data" / "sample" / "clients_sample.parquet"

modele = mlflow.pyfunc.load_model(str(ROOT / "model"))
colonnes = modele.metadata.get_input_schema().input_names()

clients = pd.read_parquet(SOURCE, columns=["SK_ID_CURR", *colonnes])
echantillon = clients.head(1000)

SORTIE.parent.mkdir(parents=True, exist_ok=True)
echantillon.to_parquet(SORTIE, index=False)
taille_mo = SORTIE.stat().st_size / 1e6
print(f"{len(echantillon)} clients → {SORTIE} ({taille_mo:.1f} Mo)")