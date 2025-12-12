#!/usr/bin/env python
"""Test script to diagnose import issues."""
import sys
import traceback

print("Python version:", sys.version)
print("Python path:", sys.path)
print()

# Test basic import
try:
    import vla_training
    print("✓ vla_training imported")
    print("  Location:", vla_training.__file__)
except Exception as e:
    print("✗ Failed to import vla_training:")
    traceback.print_exc()
    sys.exit(1)

# Test data module import
try:
    import vla_training.data
    print("✓ vla_training.data imported")
    print("  Location:", vla_training.data.__file__)
except Exception as e:
    print("✗ Failed to import vla_training.data:")
    traceback.print_exc()
    print("\nTrying to import dependencies...")
    
    # Check dependencies
    deps = ['yaml', 'pydantic', 'numpy', 'pandas']
    for dep in deps:
        try:
            __import__(dep)
            print(f"  ✓ {dep} available")
        except ImportError:
            print(f"  ✗ {dep} MISSING")
    
    # Try importing with explicit path
    print("\nTrying with explicit src path...")
    import os
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src_path = os.path.join(repo_root, 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    try:
        import vla_training.data
        print("✓ vla_training.data imported with explicit src path")
        print("  Location:", vla_training.data.__file__)
    except Exception as e2:
        print("✗ Still failed with explicit path:")
        traceback.print_exc()
        sys.exit(1)

# Test specific imports
try:
    from vla_training.data import Validator, load_spec
    print("✓ Validator and load_spec imported successfully")
except Exception as e:
    print("✗ Failed to import Validator, load_spec:")
    traceback.print_exc()
    sys.exit(1)

print("\nAll imports successful!")

