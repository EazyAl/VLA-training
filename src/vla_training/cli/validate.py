from __future__ import annotations

import argparse
import sys
from pathlib import Path

from vla_training.data import Validator, load_spec


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate dataset against a DatasetSpec.")
    parser.add_argument("--data-root", required=True, type=Path, help="Dataset root directory.")
    parser.add_argument("--spec", required=True, type=Path, help="Path to dataset spec (yaml/json).")
    parser.add_argument("--report", type=Path, help="Optional path to write JSON report.")
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Optional limit on number of episodes to validate.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    spec = load_spec(args.spec)
    validator = Validator(spec=spec, dataset_root=args.data_root)
    report = validator.validate(max_episodes=args.max_episodes)

    if args.report:
        report.write_json(args.report)
        print(f"Wrote report to {args.report}")

    print(
        f"Checked {report.episodes_checked} episode(s); failures: {report.episodes_failed}",
        file=sys.stdout,
    )
    if report.episodes_failed:
        for ep in report.episode_issues:
            if not ep.issues:
                continue
            print(f"[{ep.episode}]")
            for issue in ep.issues:
                print(f"  - {issue}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

