from __future__ import annotations

import argparse
from pathlib import Path

from vla_training.data import DatasetSpec, compute_norms, load_spec


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute normalization stats for a dataset.")
    parser.add_argument("--data-root", required=True, type=Path, help="Dataset root directory.")
    parser.add_argument("--spec", required=True, type=Path, help="Path to dataset spec (yaml/json).")
    parser.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="Directory to write norms.npz and metadata.json.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Optional limit on number of episodes to use for stats.",
    )
    parser.add_argument(
        "--proprio-keys",
        nargs="*",
        default=None,
        help="Optional list of proprio keys to normalize (subset of spec.required_proprio_keys).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    spec: DatasetSpec = load_spec(args.spec)
    proprio_keys = args.proprio_keys or spec.required_proprio_keys
    norms = compute_norms(
        spec=spec,
        dataset_root=args.data_root,
        out_dir=args.out_dir,
        max_episodes=args.max_episodes,
        proprio_keys=proprio_keys,
    )
    print(f"Saved norms to {args.out_dir}")
    print(f"Action mean/std shapes: {norms.action_mean.shape}/{norms.action_std.shape}")
    if norms.proprio_mean:
        keys = ', '.join(norms.proprio_mean.keys())
        print(f"Proprio keys normalized: {keys}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

