"""Tests du modèle : structure, non-régression, seuil métier, reproductibilité."""
import pytest

from src.predict import OBLIGATOIRES, predire, predire_client, preparer_features

SEUIL = 0.08


def test_modele_attend_796_features(types):
    assert len(types) == 796


def test_code_gender_absent(types):
    """La v2 a retiré CODE_GENDER (équité) : il ne doit pas revenir."""
    assert not any(col.startswith("CODE_GENDER") for col in types)


def test_non_regression_client_100001(modele, clients, types):
    resultat = predire_client(modele, clients, types, 100001)
    assert resultat["probabilite_defaut"] == pytest.approx(0.0500, abs=1e-3)
    assert resultat["decision"] == "accordé"


def test_probabilites_et_decisions_coherentes_avec_le_seuil(modele, clients, types):
    sortie = modele.predict(clients[list(types)].head(200))
    proba = sortie["probabilite_defaut"]
    refus = sortie["decision"] == "refusé"

    assert proba.between(0, 1).all()
    assert set(sortie["decision"]) <= {"accordé", "refusé"}
    loin_du_seuil = (proba - SEUIL).abs() > 1e-6
    assert ((proba > SEUIL) == refus)[loin_du_seuil].all()


def test_reproductibilite(modele, clients, types):
    premiere = predire_client(modele, clients, types, 100001)
    seconde = predire_client(modele, clients, types, 100001)
    assert premiere == seconde


def test_prediction_avec_seulement_les_champs_obligatoires(modele, types, features_valides):
    """Tolérance aux valeurs manquantes : LightGBM gère les NaN nativement."""
    minimal = {k: features_valides[k] for k in OBLIGATOIRES}
    resultat = predire(modele, preparer_features(minimal, types))
    assert 0 <= resultat["probabilite_defaut"] <= 1