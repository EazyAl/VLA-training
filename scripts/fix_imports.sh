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
echo "Contents: $(cat $PTH_FILE)"

# Test if the path exists
if [ ! -d "$REPO_ROOT/src" ]; then
    echo "ERROR: $REPO_ROOT/src does not exist!"
    exit 1
fi

# Test if vla_training exists in src
if [ ! -d "$REPO_ROOT/src/vla_training" ]; then
    echo "ERROR: $REPO_ROOT/src/vla_training does not exist!"
    exit 1
fi

# Test import with explicit path addition
echo "Testing import with explicit path..."
python -c "
import sys
sys.path.insert(0, '$REPO_ROOT/src')
try:
    from vla_training.data import Validator, load_spec
    print('✓ Imports working with explicit path!')
except Exception as e:
    print(f'✗ Import failed: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
"

# Now test with .pth file (may need new Python process)
echo ""
echo "Testing with .pth file (new Python process)..."
python -c "
import sys
print('Python path includes:')
for p in sys.path:
    if 'VLA-training' in p or 'vla' in p.lower():
        print(f'  {p}')

try:
    from vla_training.data import Validator, load_spec
    print('✓ Imports working with .pth file!')
except Exception as e:
    print(f'✗ Import failed: {e}')
    print('Note: You may need to start a new Python session for .pth to take effect')
    import traceback
    traceback.print_exc()
"

echo ""
echo "Done! If imports still fail, try:"
echo "  1. Start a new Python session (the .pth file is now in place)"
echo "  2. Or use: export PYTHONPATH=\"$REPO_ROOT/src:\$PYTHONPATH\""

