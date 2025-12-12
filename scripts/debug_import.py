#!/usr/bin/env python
"""Debug script to find the exact import issue."""
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path("/workspace/VLA-training")
SRC_PATH = REPO_ROOT / "src"

print(f"Adding {SRC_PATH} to path...")
sys.path.insert(0, str(SRC_PATH))

print(f"\nPython path now includes:")
for p in sys.path[:5]:
    print(f"  {p}")

# Test 1: Can we import vla_training?
print("\n[Test 1] Importing vla_training...")
try:
    import vla_training
    print(f"✓ vla_training imported from: {vla_training.__file__}")
    print(f"  __path__: {vla_training.__path__}")
except Exception as e:
    print(f"✗ Failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 2: Does the data directory exist?
print("\n[Test 2] Checking data directory...")
data_dir = Path(vla_training.__file__).parent / "data"
print(f"  Expected path: {data_dir}")
print(f"  Exists: {data_dir.exists()}")
print(f"  Is directory: {data_dir.is_dir()}")
if data_dir.exists():
    init_file = data_dir / "__init__.py"
    print(f"  __init__.py exists: {init_file.exists()}")

# Test 3: Try importing data module directly
print("\n[Test 3] Importing vla_training.data...")
try:
    import vla_training.data
    print(f"✓ vla_training.data imported from: {vla_training.data.__file__}")
except Exception as e:
    print(f"✗ Failed: {e}")
    print("\nFull traceback:")
    traceback.print_exc()
    
    # Try to see what's in the data directory
    print("\nFiles in data directory:")
    if data_dir.exists():
        for f in sorted(data_dir.iterdir()):
            print(f"  {f.name}")
    
    sys.exit(1)

# Test 4: Try importing specific items
print("\n[Test 4] Importing Validator and load_spec...")
try:
    from vla_training.data import Validator, load_spec
    print("✓ Validator and load_spec imported successfully!")
except Exception as e:
    print(f"✗ Failed: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\n✓ All imports successful!")

