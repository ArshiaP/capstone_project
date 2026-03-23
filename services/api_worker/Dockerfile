FROM python:3.11-slim

WORKDIR /app

# Install dependencies for both the API and the background worker.
COPY services/api/requirements.txt ./requirements-api.txt
COPY services/worker/requirements.txt ./requirements-worker.txt
RUN pip install --no-cache-dir -r requirements-api.txt -r requirements-worker.txt

# App code
COPY services/api/main.py ./main.py
COPY services/worker/worker.py ./worker.py

# Seed (pretrained models + metadata). Worker expects these files under $ARTIFACT_DIR.
RUN mkdir -p /seed_artifacts
# The repo historically used a typo’d filename for the GaussianCopula model.
COPY artifacts/gaussian_copuula_diabetes.pkl /seed_artifacts/
COPY artifacts/ctgan_diabetes.pkl /seed_artifacts/
COPY artifacts/tvae_diabetes.pkl /seed_artifacts/
COPY artifacts/diabetes_metadata.json /seed_artifacts/

COPY services/api_worker/start.sh ./start.sh
RUN chmod +x ./start.sh

EXPOSE 10000

CMD ["./start.sh"]

