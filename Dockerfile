# Même version de Python que l'entraînement (le wrapper CloudPickle en dépend)
FROM python:3.11.9-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

# libgomp1 : bibliothèque système nécessaire à LightGBM
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dépendances d'abord : cette couche reste en cache tant que requirements.txt ne change pas
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Uniquement ce qui sert à l'API
COPY api/ api/
COPY src/ src/
COPY model/ model/
COPY data/sample/ data/sample/

# Exécution sans les droits root (sécurité)
RUN useradd --create-home appuser
USER appuser

EXPOSE 8080

# Cloud Run fournit le port via la variable PORT
CMD exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT}