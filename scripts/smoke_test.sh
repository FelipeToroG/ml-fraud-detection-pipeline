#!/usr/bin/env bash
# scripts/smoke_test.sh
# =====================
# Manual end-to-end check of the API outside of pytest.
#
# Usage:
#   1. Start the service in a separate terminal:
#        uvicorn src.api.main:app --reload --port 8000
#   2. Run this script:
#        bash scripts/smoke_test.sh
#
# Confirms the service is alive, the model is loaded, and a representative
# transaction returns a well-formed prediction.

set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"

echo "==> GET /health"
curl -fsS "${BASE_URL}/health" | python -m json.tool

echo
echo "==> POST /predict"
curl -fsS -X POST "${BASE_URL}/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "X1":  0.0, "X2":  0.0, "X3":  0.0, "X4":  0.0, "X5":  0.0,
    "X6":  0.0, "X7":  0.0, "X8":  0.0, "X9":  0.0, "X10": 0.0,
    "X11": 0.0, "X12": 0.0, "X13": 0.0, "X14": 0.0, "X15": 0.0,
    "X16": 0.0, "X17": 0.0, "X18": 0.0, "X19": 0.0, "X20": 0.0,
    "X21": 0.0, "X22": 0.0, "X23": 0.0, "X24": 0.0, "X25": 0.0,
    "X26": 0.0, "X27": 0.0, "X28": 0.0,
    "X29": 100.00,
    "X30": 86400.0
  }' | python -m json.tool

echo
echo "Smoke test completed."
