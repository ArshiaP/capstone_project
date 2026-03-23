#!/bin/sh
set -eu

ARTIFACT_DIR="${ARTIFACT_DIR:-/artifacts}"
PORT="${PORT:-10000}"

mkdir -p "$ARTIFACT_DIR"

# If the Render disk mount is empty (first deploy) or has the wrong seed files,
# copy the pretrained models/metadata from the image into $ARTIFACT_DIR.
GAUSSIAN_COPULA_PATH="$ARTIFACT_DIR/gaussian_copula_diabetes.pkl"
GAUSSIAN_COPUULA_PATH="$ARTIFACT_DIR/gaussian_copuula_diabetes.pkl"

# Note: we consider "Gaussian" seeded if either filename exists.
GAUSSIANSEEDED=false
if [ -f "$GAUSSIAN_COPULA_PATH" ] || [ -f "$GAUSSIAN_COPUULA_PATH" ]; then
  GAUSSIANSEEDED=true
fi

if [ "$GAUSSIANSEEDED" != "true" ] || \
   [ ! -f "$ARTIFACT_DIR/ctgan_diabetes.pkl" ] || \
   [ ! -f "$ARTIFACT_DIR/tvae_diabetes.pkl" ] || \
   [ ! -f "$ARTIFACT_DIR/diabetes_metadata.json" ]; then
  # Copy whichever Gaussian filename is available in the image.
  if [ -f "/seed_artifacts/gaussian_copula_diabetes.pkl" ]; then
    cp -f /seed_artifacts/gaussian_copula_diabetes.pkl "$ARTIFACT_DIR/" || true
  fi
  if [ -f "/seed_artifacts/gaussian_copuula_diabetes.pkl" ]; then
    cp -f /seed_artifacts/gaussian_copuula_diabetes.pkl "$ARTIFACT_DIR/" || true
  fi
  cp -f /seed_artifacts/ctgan_diabetes.pkl "$ARTIFACT_DIR/" || true
  cp -f /seed_artifacts/tvae_diabetes.pkl "$ARTIFACT_DIR/" || true
  cp -f /seed_artifacts/diabetes_metadata.json "$ARTIFACT_DIR/" || true
fi

terminate() {
  # Best-effort: stop both processes on container shutdown.
  if [ "${WORKER_PID:-}" != "" ]; then
    kill "$WORKER_PID" 2>/dev/null || true
  fi
  if [ "${API_PID:-}" != "" ]; then
    kill "$API_PID" 2>/dev/null || true
  fi
}
trap terminate INT TERM

python -u /app/worker.py &
WORKER_PID=$!

uvicorn main:app --host 0.0.0.0 --port "$PORT" &
API_PID=$!

wait "$API_PID"

