"""API de scoring crédit – Prêt à dépenser.

Lancement local : uvicorn api.main:app --reload --port 8000
Documentation interactive : http://127.0.0.1:8000/docs
"""
import logging
import os
import secrets
import time
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader

from api.schemas import DemandeClient, DemandeFeatures, Prediction
from src.predict import (
    charger_clients,
    charger_modele,
    predire,
    predire_client,
    preparer_features,
    types_attendus,
    valider_features,
)

load_dotenv()  # lit .env en local ; sur Cloud Run, les variables viennent de l'environnement

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("api")

# Ressources chargées UNE SEULE FOIS au démarrage, réutilisées par toutes les requêtes
etat = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    debut = time.perf_counter()
    etat["modele"] = charger_modele()
    etat["types"] = types_attendus(etat["modele"])
    etat["clients"] = charger_clients()
    logger.info(
        "Modèle chargé (%d features, %d clients) en %.1f s",
        len(etat["types"]), len(etat["clients"]), time.perf_counter() - debut,
    )
    yield
    etat.clear()


app = FastAPI(
    title="API Scoring Crédit – Prêt à dépenser",
    description=(
        "Prédit la probabilité de défaut d'un client et décide si le crédit est "
        "accordé ou refusé (LightGBM, seuil métier 0,08). Les routes de prédiction "
        "nécessitent une clé API (en-tête X-API-Key)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------- Gestion des erreurs et sécurité ----------

@app.exception_handler(Exception)
async def erreur_inattendue(request: Request, exc: Exception):
    """Toute erreur non prévue renvoie un 500 propre, sans exposer la trace."""
    logger.exception("Erreur inattendue sur %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Erreur interne du serveur"})


def verifier_modele():
    if "modele" not in etat:
        raise HTTPException(status_code=503, detail="Modèle non chargé")


cle_api_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verifier_cle_api(cle: str | None = Security(cle_api_header)):
    """Protège les routes de prédiction : 401 sans clé, 403 si clé incorrecte."""
    attendue = os.getenv("API_KEY")
    if not attendue:
        raise HTTPException(status_code=500, detail="Clé API non configurée sur le serveur")
    if cle is None:
        raise HTTPException(status_code=401, detail="Clé API manquante (en-tête X-API-Key)")
    if not secrets.compare_digest(cle, attendue):
        raise HTTPException(status_code=403, detail="Clé API invalide")


ERREURS_AUTH = {
    401: {"description": "Clé API manquante"},
    403: {"description": "Clé API invalide"},
}


# ---------- Routes ----------

@app.get("/", summary="Accueil")
def accueil():
    return {"message": "API Scoring Crédit – Prêt à dépenser", "documentation": "/docs"}


@app.get("/health", summary="État de l'API", responses={503: {"description": "Modèle non chargé"}})
def health():
    """Vérifie que l'API tourne et que le modèle est chargé (route publique)."""
    verifier_modele()
    return {
        "status": "ok",
        "nb_features": len(etat["types"]),
        "nb_clients_disponibles": len(etat["clients"]),
    }


@app.post(
    "/predict/client",
    response_model=Prediction,
    summary="Score d'un client connu",
    dependencies=[Depends(verifier_cle_api)],
    responses={
        **ERREURS_AUTH,
        404: {"description": "Client introuvable"},
        422: {"description": "Entrée invalide"},
    },
)
def predict_client(demande: DemandeClient):
    """Calcule le score d'un client à partir de son SK_ID_CURR (features déjà préparées)."""
    verifier_modele()
    debut = time.perf_counter()
    try:
        resultat = predire_client(etat["modele"], etat["clients"], etat["types"], demande.sk_id_curr)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Client {demande.sk_id_curr} introuvable")
    duree_ms = round((time.perf_counter() - debut) * 1000, 2)
    return Prediction(sk_id_curr=demande.sk_id_curr, temps_inference_ms=duree_ms, **resultat)


@app.post(
    "/predict",
    response_model=Prediction,
    summary="Score à partir des features",
    dependencies=[Depends(verifier_cle_api)],
    responses={
        **ERREURS_AUTH,
        422: {"description": "Champ obligatoire manquant, valeur hors plage ou type incorrect"},
    },
)
def predict_features(demande: DemandeFeatures):
    """Calcule le score à partir des features d'un client (nouvelle demande)."""
    verifier_modele()
    erreurs = valider_features(demande.features, etat["types"])
    if erreurs:
        raise HTTPException(status_code=422, detail=erreurs)
    debut = time.perf_counter()
    X = preparer_features(demande.features, etat["types"])
    resultat = predire(etat["modele"], X)
    duree_ms = round((time.perf_counter() - debut) * 1000, 2)
    return Prediction(temps_inference_ms=duree_ms, **resultat)