#!/usr/bin/env python
"""Check what files actually exist in the repository."""
from pathlib import Path
import os

REPO_ROOT = Path("/workspace/VLA-training")
SRC_DIR = REPO_ROOT / "src"
VLA_DIR = SRC_DIR / "vla_training"

print(f"Repository root: {REPO_ROOT}")
print(f"Exists: {REPO_ROOT.exists()}")
print()

print(f"Source directory: {SRC_DIR}")
print(f"Exists: {SRC_DIR.exists()}")
print()

print(f"VLA training directory: {VLA_DIR}")
print(f"Exists: {VLA_DIR.exists()}")
print()

if VLA_DIR.exists():
    print("Contents of vla_training:")
    for item in sorted(VLA_DIR.iterdir()):
        if item.is_dir():
            print(f"  📁 {item.name}/")
            # List contents of subdirectories
            try:
                for subitem in sorted(item.iterdir()):
                    marker = "📁" if subitem.is_dir() else "📄"
                    print(f"      {marker} {subitem.name}")
            except PermissionError:
                print(f"      (permission denied)")
        else:
            print(f"  📄 {item.name}")
else:
    print("ERROR: vla_training directory does not exist!")
    print(f"\nContents of {SRC_DIR}:")
    if SRC_DIR.exists():
        for item in sorted(SRC_DIR.iterdir()):
            marker = "📁" if item.is_dir() else "📄"
            print(f"  {marker} {item.name}")
    else:
        print("  (does not exist)")

