#!/usr/bin/env python
from __future__ import annotations

import platform
import sys


def main() -> None:
    print(f"Python: {sys.version.split()[0]} ({platform.system()} {platform.machine()})")

    try:
        import torch
    except Exception as exc:  # pragma: no cover - runtime diagnostic
        print(f"Torch: not installed ({exc})")
    else:
        print(f"Torch: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA devices: {torch.cuda.device_count()}")
            for idx in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(idx)
                print(f"    [{idx}] {props.name} (cap {props.major}.{props.minor})")

        rocm_available = getattr(torch.version, "hip", None) is not None
        print(f"  ROCm available: {rocm_available}")
        if rocm_available:
            print(f"  ROCm version: {torch.version.hip}")

    try:
        import lerobot
    except Exception as exc:  # pragma: no cover - runtime diagnostic
        print(f"LeRobot: not installed ({exc})")
    else:
        print(f"LeRobot: {lerobot.__version__}")


if __name__ == "__main__":
    main()

