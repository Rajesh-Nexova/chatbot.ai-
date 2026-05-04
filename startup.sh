#!/bin/bash

echo "===== STARTUP DEBUG ====="
echo "PORT=$PORT"
echo "PWD=$(pwd)"
ls -la

echo "Starting FastAPI with uvicorn..."

exec python -m uvicorn app.main:app \
  --host 0.0.0.0 \
  --port $PORT \
  --log-level info