"""Formats d'entrée et de sortie de l'API (validés automatiquement par Pydantic)."""
from typing import Optional

from pydantic import BaseModel, Field, StrictFloat


class DemandeClient(BaseModel):
    sk_id_curr: int = Field(
        ..., gt=0, description="Identifiant du client (SK_ID_CURR)", examples=[100001]
    )


class DemandeFeatures(BaseModel):
    features: dict[str, Optional[StrictFloat]] = Field(
        ...,
        description=(
            "Features du client (noms des colonnes du modèle). Les features absentes "
            "sont traitées comme manquantes. Obligatoires : AMT_INCOME_TOTAL, "
            "AMT_CREDIT, AMT_ANNUITY, DAYS_BIRTH."
        ),
        examples=[{
            "AMT_INCOME_TOTAL": 135000,
            "AMT_CREDIT": 568800,
            "AMT_ANNUITY": 20560.5,
            "DAYS_BIRTH": -19241,
            "EXT_SOURCE_2": 0.79,
            "EXT_SOURCE_3": 0.16,
        }],
    )


class Prediction(BaseModel):
    sk_id_curr: Optional[int] = Field(None, description="Identifiant du client, si connu")
    probabilite_defaut: float = Field(..., description="Probabilité de défaut (0 à 1)")
    decision: str = Field(..., description="« accordé » ou « refusé » (seuil métier 0,08)")
    temps_inference_ms: float = Field(..., description="Temps de calcul de la prédiction")