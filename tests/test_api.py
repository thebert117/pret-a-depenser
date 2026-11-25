"""Tests fonctionnels : l'API appelée comme le ferait Crédit Express."""
import pytest
from fastapi.testclient import TestClient

import api.main
from api.main import app
from src.predict import OBLIGATOIRES


# ---------- Routes publiques ----------

def test_accueil(client):
    reponse = client.get("/")
    assert reponse.status_code == 200
    assert reponse.json()["documentation"] == "/docs"


def test_health_sans_cle(client):
    reponse = client.get("/health")
    assert reponse.status_code == 200
    assert reponse.json()["status"] == "ok"
    assert reponse.json()["nb_features"] == 796


# ---------- Authentification ----------

def test_sans_cle_api_401(client):
    reponse = client.post("/predict/client", json={"sk_id_curr": 100001})
    assert reponse.status_code == 401


def test_mauvaise_cle_api_403(client):
    reponse = client.post("/predict/client", json={"sk_id_curr": 100001}, headers={"X-API-Key": "fausse"})
    assert reponse.status_code == 403


def test_cle_api_non_configuree_500(client, headers, monkeypatch):
    monkeypatch.delenv("API_KEY")
    reponse = client.post("/predict/client", json={"sk_id_curr": 100001}, headers=headers)
    assert reponse.status_code == 500


# ---------- /predict/client ----------

def test_predict_client_ok(client, headers):
    reponse = client.post("/predict/client", json={"sk_id_curr": 100001}, headers=headers)
    assert reponse.status_code == 200
    corps = reponse.json()
    assert corps["sk_id_curr"] == 100001
    assert 0 <= corps["probabilite_defaut"] <= 1
    assert corps["decision"] in {"accordé", "refusé"}
    assert corps["temps_inference_ms"] >= 0


def test_client_introuvable_404(client, headers):
    reponse = client.post("/predict/client", json={"sk_id_curr": 999}, headers=headers)
    assert reponse.status_code == 404


@pytest.mark.parametrize("valeur", ["abc", -5, 0, None])
def test_identifiant_invalide_422(client, headers, valeur):
    reponse = client.post("/predict/client", json={"sk_id_curr": valeur}, headers=headers)
    assert reponse.status_code == 422


def test_corps_vide_422(client, headers):
    reponse = client.post("/predict/client", json={}, headers=headers)
    assert reponse.status_code == 422


# ---------- /predict ----------

def test_predict_features_ok(client, headers, features_valides):
    reponse = client.post("/predict", json={"features": features_valides}, headers=headers)
    assert reponse.status_code == 200
    assert reponse.json()["sk_id_curr"] is None
    assert 0 <= reponse.json()["probabilite_defaut"] <= 1


@pytest.mark.parametrize("champ", OBLIGATOIRES)
def test_champ_obligatoire_manquant_422(client, headers, features_valides, champ):
    del features_valides[champ]
    reponse = client.post("/predict", json={"features": features_valides}, headers=headers)
    assert reponse.status_code == 422


@pytest.mark.parametrize("champ,valeur", [
    ("AMT_INCOME_TOTAL", 0),
    ("AMT_CREDIT", -1000),
    ("EXT_SOURCE_2", 1.5),
    ("DAYS_BIRTH", -1000),
])
def test_valeur_hors_plage_422(client, headers, features_valides, champ, valeur):
    features_valides[champ] = valeur
    reponse = client.post("/predict", json={"features": features_valides}, headers=headers)
    assert reponse.status_code == 422


def test_texte_au_lieu_dun_nombre_422(client, headers, features_valides):
    features_valides["EXT_SOURCE_2"] = "abc"
    reponse = client.post("/predict", json={"features": features_valides}, headers=headers)
    assert reponse.status_code == 422


def test_feature_inconnue_422(client, headers, features_valides):
    features_valides["COLONNE_INVENTEE"] = 1
    reponse = client.post("/predict", json={"features": features_valides}, headers=headers)
    assert reponse.status_code == 422


# ---------- Robustesse ----------

def test_modele_non_charge_503(client, monkeypatch):
    monkeypatch.delitem(api.main.etat, "modele")
    assert client.get("/health").status_code == 503


def test_erreur_inattendue_500_sans_trace(client, headers, monkeypatch):
    def panne(*args, **kwargs):
        raise RuntimeError("panne simulée")

    monkeypatch.setattr(api.main, "predire_client", panne)
    client_sans_exception = TestClient(app, raise_server_exceptions=False)
    reponse = client_sans_exception.post("/predict/client", json={"sk_id_curr": 100001}, headers=headers)
    assert reponse.status_code == 500
    assert reponse.json() == {"detail": "Erreur interne du serveur"}