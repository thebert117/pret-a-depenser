"""Tests unitaires : validation et préparation des features, sans le vrai modèle."""
import numpy as np
import pytest

from src.predict import OBLIGATOIRES, est_colonne_discrete, preparer_features, valider_features

# Signature simplifiée : 5 colonnes décimales + 1 colonne entière (type FLAG)
TYPES = {
    "AMT_INCOME_TOTAL": np.dtype("float32"),
    "AMT_CREDIT": np.dtype("float32"),
    "AMT_ANNUITY": np.dtype("float32"),
    "DAYS_BIRTH": np.dtype("float32"),
    "EXT_SOURCE_2": np.dtype("float32"),
    "FLAG_OWN_CAR": np.dtype("int64"),
}
VALIDES = {"AMT_INCOME_TOTAL": 135000, "AMT_CREDIT": 568800, "AMT_ANNUITY": 20560.5, "DAYS_BIRTH": -19241}


def test_features_valides_sans_erreur():
    assert valider_features(VALIDES, TYPES) == []


@pytest.mark.parametrize("champ", OBLIGATOIRES)
def test_champ_obligatoire_absent(champ):
    features = {k: v for k, v in VALIDES.items() if k != champ}
    erreurs = valider_features(features, TYPES)
    assert any(champ in e for e in erreurs)


def test_champ_obligatoire_a_null():
    erreurs = valider_features({**VALIDES, "AMT_CREDIT": None}, TYPES)
    assert any("AMT_CREDIT" in e for e in erreurs)


@pytest.mark.parametrize("champ,valeur", [
    ("AMT_INCOME_TOTAL", 0),     # revenu nul
    ("EXT_SOURCE_2", 1.5),       # score externe > 1
    ("EXT_SOURCE_2", -0.1),      # score externe < 0
    ("DAYS_BIRTH", -1000),       # moins de 18 ans
    ("DAYS_BIRTH", 100),         # date de naissance dans le futur
])
def test_valeur_hors_plage(champ, valeur):
    erreurs = valider_features({**VALIDES, champ: valeur}, TYPES)
    assert any(champ in e and "hors de la plage" in e for e in erreurs)


@pytest.mark.parametrize("valeur", [0, 1])
def test_bornes_incluses(valeur):
    assert valider_features({**VALIDES, "EXT_SOURCE_2": valeur}, TYPES) == []


def test_feature_inconnue():
    erreurs = valider_features({**VALIDES, "COLONNE_INVENTEE": 1}, TYPES)
    assert any("COLONNE_INVENTEE" in e for e in erreurs)


def test_entier_attendu_sur_colonne_discrete():
    assert valider_features({**VALIDES, "FLAG_OWN_CAR": 1.0}, TYPES) == []
    erreurs = valider_features({**VALIDES, "FLAG_OWN_CAR": 0.5}, TYPES)
    assert any("FLAG_OWN_CAR" in e and "entier" in e for e in erreurs)


@pytest.mark.parametrize("dtype,attendu", [
    (np.dtype("float32"), False),
    (np.dtype("float64"), False),
    (np.dtype("int64"), True),
    (np.dtype("bool"), True),
])
def test_est_colonne_discrete(dtype, attendu):
    assert est_colonne_discrete(dtype) is attendu


def test_preparer_features_complete_les_manquantes():
    X = preparer_features(VALIDES, TYPES)
    assert X.shape == (1, len(TYPES))
    assert list(X.columns) == list(TYPES)            # ordre de la signature respecté
    assert np.isnan(X.loc[0, "EXT_SOURCE_2"])        # décimale absente -> NaN
    assert X.loc[0, "FLAG_OWN_CAR"] == 0             # discrète absente -> 0
    assert X.dtypes.to_dict() == TYPES               # types de la signature respectés