#!/usr/bin/env python
"""
Download dataset and cache models from HuggingFace for remote GPU setup.
Assumes environment (PyTorch, ROCm, requirements) is already set up.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
)


def download_dataset(
    repo_id: str = "PRFitz/lekiwi-dataset-pick-place-red123",
    out_dir: Path = Path("data/lekiwi_pickplace"),
    max_episodes: int = 50,
) -> None:
    """Download and prepare dataset from HuggingFace."""
    print(f"[1/3] Downloading dataset from {repo_id}...")
    
    # Call prepare_lekiwi script as subprocess
    import subprocess
    import sys
    
    scripts_dir = Path(__file__).parent
    prepare_script = scripts_dir / "prepare_lekiwi.py"
    
    cmd = [
        sys.executable,
        str(prepare_script),
        "--out-dir", str(out_dir),
        "--repo-id", repo_id,
        "--max-episodes", str(max_episodes),
    ]
    
    result = subprocess.run(cmd, check=True)
    print(f"✓ Dataset prepared in {out_dir}")


def cache_lerobot_models(pi05_model: str = "lerobot/pi05_base") -> None:
    """Cache Pi05 policy models from HuggingFace.
    
    Downloads the Pi05 policy checkpoint so it doesn't need to be downloaded
    on every training run. Available models:
    - lerobot/pi05_base: Base Pi05 model
    - lerobot/pi05_libero: Pi05 model trained on Libero dataset
    
    Reference: https://huggingface.co/docs/lerobot/pi05
    """
    print("\n[2/3] Caching Pi05 policy models from HuggingFace...")
    
    cache_dir = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    print(f"Cache directory: {cache_dir}")
    print(f"  (Set HF_HOME env var to change location)")
    
    # Pi05 policy models to cache
    pi05_models = [
        "lerobot/pi05_base",      # Base Pi05 model (recommended default)
        "lerobot/pi05_libero",    # Libero-trained variant
    ]
    
    # Also cache the specified model if it's different
    if pi05_model not in pi05_models:
        pi05_models.append(pi05_model)
    
    for model_id in pi05_models:
        try:
            print(f"  Caching Pi05 policy: {model_id}...")
            # Download entire repository snapshot (includes config, weights, etc.)
            snapshot_download(
                repo_id=model_id,
                cache_dir=cache_dir,
                repo_type="model",
            )
            print(f"    ✓ Cached {model_id}")
        except Exception as e:
            print(f"    ⚠ Warning: Could not cache {model_id}: {e}")
            print(f"      (This may cause downloads during training if model is needed)")
    
    # Also cache underlying transformer models that Pi05 uses
    print("\n  Caching underlying transformer models used by Pi05...")
    transformer_models = [
        # SigLIP vision encoder (used by Pi05)
        ("google/siglip-base-patch16-224", "AutoModel", "AutoProcessor"),
        # Phi language model (used by Pi05)
        ("microsoft/phi-1.5", "AutoModelForCausalLM", "AutoTokenizer"),
    ]
    
    for model_id, model_class, tokenizer_class in transformer_models:
        try:
            print(f"    Caching {model_id}...")
            
            # Download tokenizer/processor
            if tokenizer_class == "AutoTokenizer":
                AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
            elif tokenizer_class == "AutoProcessor":
                AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)
            
            # Download model
            if model_class == "AutoModelForCausalLM":
                AutoModelForCausalLM.from_pretrained(
                    model_id,
                    cache_dir=cache_dir,
                    torch_dtype="auto",
                )
            else:
                AutoModel.from_pretrained(
                    model_id,
                    cache_dir=cache_dir,
                    torch_dtype="auto",
                )
            print(f"      ✓ Cached {model_id}")
        except Exception as e:
            print(f"      ⚠ Warning: Could not cache {model_id}: {e}")
            print(f"        (This is okay if the model isn't needed)")


def verify_setup(data_dir: Path) -> None:
    """Verify that dataset and required files exist."""
    print("\n[3/3] Verifying setup...")
    
    # Check episodes exist
    ep_dir = data_dir / "episodes"
    if not ep_dir.exists():
        raise FileNotFoundError(f"Episodes directory not found: {ep_dir}")
    
    episodes = list(ep_dir.glob("*.npz"))
    if not episodes:
        raise FileNotFoundError(f"No episodes found in {ep_dir}")
    
    print(f"✓ Found {len(episodes)} episodes in {ep_dir}")
    
    # Check meta file
    meta_file = data_dir / "prepared_meta.json"
    if meta_file.exists():
        print(f"✓ Dataset metadata found: {meta_file}")
    else:
        print(f"⚠ Dataset metadata not found (optional): {meta_file}")
    
    print("\n✓ Setup verification complete!")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download dataset and cache models from HuggingFace for remote GPU setup"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/lekiwi_pickplace"),
        help="Directory to store dataset",
    )
    parser.add_argument(
        "--dataset-repo",
        default="PRFitz/lekiwi-dataset-pick-place-red123",
        help="HuggingFace dataset repository ID",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=50,
        help="Maximum number of episodes to prepare",
    )
    parser.add_argument(
        "--skip-dataset",
        action="store_true",
        help="Skip dataset download (if already exists)",
    )
    parser.add_argument(
        "--skip-models",
        action="store_true",
        help="Skip model caching",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="HuggingFace cache directory (default: ~/.cache/huggingface)",
    )
    parser.add_argument(
        "--pi05-model",
        default="lerobot/pi05_base",
        help="Pi05 policy model to cache (default: lerobot/pi05_base). "
             "Options: lerobot/pi05_base, lerobot/pi05_libero",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Dataset & Model Cache Setup for Remote GPU")
    print("=" * 60)
    print()

    # Set cache directory if provided
    if args.cache_dir:
        os.environ["HF_HOME"] = str(args.cache_dir)
        print(f"Using cache directory: {args.cache_dir}")

    # Download dataset
    if not args.skip_dataset:
        download_dataset(
            repo_id=args.dataset_repo,
            out_dir=args.data_dir,
            max_episodes=args.max_episodes,
        )
    else:
        print("[1/3] Skipping dataset download (--skip-dataset)")

    # Cache models
    if not args.skip_models:
        cache_lerobot_models(pi05_model=args.pi05_model)
    else:
        print("\n[2/3] Skipping model caching (--skip-models)")

    # Verify setup
    if not args.skip_dataset:
        verify_setup(args.data_dir)
    else:
        print("\n[3/3] Skipping verification (--skip-dataset)")

    print()
    print("=" * 60)
    print("Setup Complete! ✓")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. Compute normalization stats:")
    print(f"     python -m vla_training.cli.compute_norms \\")
    print(f"       --data-root {args.data_dir} \\")
    print(f"       --spec configs/specs/mechalabs.yaml \\")
    print(f"       --out-dir artifacts/norms/lekiwi_pickplace")
    print()
    print("  2. Start training (models will load from cache):")
    print("     python -m vla_training.cli.train --config configs/phase_a.json")
    print()
    print("Cached models location:")
    cache_dir = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    print(f"  {cache_dir}")
    print()
    print("Note: Pi05 policy models are cached and won't re-download on subsequent runs.")
    print("      To use a different Pi05 model, specify --pi05-model when caching.")
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

