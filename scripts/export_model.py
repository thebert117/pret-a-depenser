"""Exporte le modèle champion du Model Registry MLflow vers model/.

Le dossier model/ est autonome : l'API et Docker n'ont plus besoin
de mlflow.db ni de mlruns/.
"""
import os
import shutil
from pathlib import Path

import mlflow

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)  # les chemins d'artifacts MLflow sont relatifs à la racine

mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")

TMP = ROOT / "model_tmp"
DEST = ROOT / "model"
for dossier in (TMP, DEST):
    shutil.rmtree(dossier, ignore_errors=True)

mlflow.artifacts.download_artifacts(
    artifact_uri="models:/credit_scoring_lgbm@champion",
    dst_path=str(TMP),
)

# On récupère le dossier qui contient le fichier MLmodel
dossier_modele = next(TMP.rglob("MLmodel")).parent
shutil.copytree(dossier_modele, DEST)
shutil.rmtree(TMP)

# Vérification : le modèle se recharge depuis model/
modele = mlflow.pyfunc.load_model(str(DEST))
nb_features = len(modele.metadata.get_input_schema().input_names())
print(f"Modèle exporté dans {DEST} ({nb_features} features)")