#!/usr/bin/env bash
# Simple wrapper script for dataset and model caching
# Assumes environment is already set up

set -euo pipefail

DATA_DIR="${DATA_DIR:-data/lekiwi_pickplace}"
MAX_EPISODES="${MAX_EPISODES:-50}"

echo "Downloading dataset and caching models..."
python scripts/prepare_data_and_cache.py \
    --data-dir "${DATA_DIR}" \
    --max-episodes "${MAX_EPISODES}"

echo ""
echo "Done! Next: compute norms and start training."

