"""Fixtures partagées par tous les tests."""
import os

# Clé de test définie AVANT l'import de l'app (prioritaire sur le .env local)
CLE_TEST = "cle-de-test"
os.environ["API_KEY"] = CLE_TEST

import pytest
from fastapi.testclient import TestClient

import api.main
from api.main import app

EXEMPLE_FEATURES = {
    "AMT_INCOME_TOTAL": 135000,
    "AMT_CREDIT": 568800,
    "AMT_ANNUITY": 20560.5,
    "DAYS_BIRTH": -19241,
    "EXT_SOURCE_2": 0.79,
    "EXT_SOURCE_3": 0.16,
}


@pytest.fixture(scope="session")
def client():
    """Client de test : déclenche le démarrage de l'API (chargement unique du modèle)."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def headers():
    return {"X-API-Key": CLE_TEST}


@pytest.fixture
def features_valides():
    return dict(EXEMPLE_FEATURES)


@pytest.fixture(scope="session")
def modele(client):
    return api.main.etat["modele"]


@pytest.fixture(scope="session")
def types(client):
    return api.main.etat["types"]


@pytest.fixture(scope="session")
def clients(client):
    return api.main.etat["clients"]