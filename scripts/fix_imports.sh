#!/bin/bash
# Workaround script to fix import issues with editable install

set -e

REPO_ROOT="/workspace/VLA-training"
VENV_SITE_PACKAGES="/opt/venv/lib/python3.12/site-packages"

echo "Fixing vla_training imports..."

# Create .pth file to add src to Python path
PTH_FILE="$VENV_SITE_PACKAGES/vla-training.pth"
echo "$REPO_ROOT/src" > "$PTH_FILE"
echo "Created .pth file: $PTH_FILE"

# Verify
python -c "from vla_training.data import Validator, load_spec; print('✓ Imports working!')"

echo "Done! You can now use: python -m vla_training.cli.validate"

