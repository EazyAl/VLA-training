from __future__ import annotations

import argparse
import json
from pathlib import Path

from vla_training.eval.offline import evaluate


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Offline evaluation for PI05 checkpoints.")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--spec", type=Path, required=True)
    p.add_argument("--norms", type=Path, required=True)
    p.add_argument("--max-episodes", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--mock-model", action="store_true", help="Use dummy eval path.")
    p.add_argument("--out", type=Path, default=None, help="Optional JSON output path.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    res = evaluate(
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        spec_path=args.spec,
        norms_path=args.norms,
        max_episodes=args.max_episodes,
        batch_size=args.batch_size,
        mock_model=args.mock_model,
    )
    if args.out:
        args.out.write_text(json.dumps(res.to_dict(), indent=2))
        print(f"Wrote eval to {args.out}")
    print(res.to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

