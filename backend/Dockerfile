FROM python:3.11-slim

WORKDIR /srv
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# app code, the dashboard, and the REAL model artifact (KS3) — so prod mode boots
COPY backend/app /srv/app
COPY demo /demo
COPY results/models /results/models

ENV PRAMAAN_MODEL_DIR=/results/models
EXPOSE 8000
# default command is the ingress; compose overrides for pods / verifier
CMD ["uvicorn", "app.ingress:app", "--host", "0.0.0.0", "--port", "8000"]
