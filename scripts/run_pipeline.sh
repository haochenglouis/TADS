#!/bin/bash
# Run the full TADS pipeline (Steps 1-4)
# Usage: bash scripts/run_pipeline.sh [--config configs/default.yaml]

set -e

CONFIG="${1:-configs/default.yaml}"

echo "============================================"
echo "TADS Pipeline - Full Run"
echo "Config: $CONFIG"
echo "============================================"

echo ""
echo ">>> Step 1: Generating Proxy-Labels"
python -m tads.step1 --config "$CONFIG"

echo ""
echo ">>> Step 2: Clustering and Propagation"
python -m tads.step2 --config "$CONFIG"

echo ""
echo ">>> Step 3: OOD Filtering"
python -m tads.step3 --config "$CONFIG"

echo ""
echo ">>> Step 4: Distribution Matching and Fusion"
python -m tads.step4 --config "$CONFIG"

echo ""
echo "============================================"
echo "Pipeline complete!"
echo "============================================"
