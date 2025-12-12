from __future__ import annotations

import argparse
import json
from pathlib import Path

from vla_training.train import (
    CheckpointConfig,
    DataConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainingConfig,
    TrainingRunner,
    WandbConfig,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train Pi05 with LeRobot (scaffold).")
    p.add_argument("--config", type=Path, help="Optional path to JSON config.")
    p.add_argument("--data-root", type=Path, help="Dataset root if not using --config.")
    p.add_argument("--spec", type=Path, help="Dataset spec path if not using --config.")
    p.add_argument("--norms", type=Path, help="Norms directory if not using --config.")
    p.add_argument("--total-steps", type=int, default=1000)
    p.add_argument("--seq-len", type=int, default=32)
    p.add_argument("--stride", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--checkpoint-dir", type=Path, default=Path("artifacts/checkpoints"))
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--compile-model", action="store_true")
    p.add_argument("--no-grad-checkpointing", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-level", default="INFO", help="Logging level (e.g., DEBUG, INFO).")
    p.add_argument("--mock-model", action="store_true", help="Use a tiny dummy model for fast smoke tests.")
    p.add_argument("--wandb-project", type=str, default=None, help="W&B project name to enable logging.")
    p.add_argument("--wandb-entity", type=str, default=None, help="W&B entity.")
    p.add_argument("--wandb-mode", type=str, default="disabled", choices=["online", "offline", "disabled"])
    p.add_argument("--wandb-run-name", type=str, default=None)
    p.add_argument("--wandb-tag", action="append", default=[])
    p.add_argument("--wandb-group", type=str, default=None)
    p.add_argument("--wandb-log-artifacts", action="store_true")
    return p


def parse_config(args: argparse.Namespace) -> TrainingConfig:
    if args.config:
        data = json.loads(Path(args.config).read_text())
        data_cfg = data["data"]
        cfg = TrainingConfig(
            total_steps=data["total_steps"],
            data=DataConfig(
                data_root=Path(data_cfg["data_root"]),
                spec_path=Path(data_cfg["spec_path"]),
                norms_path=Path(data_cfg["norms_path"]),
                seq_len=data_cfg.get("seq_len", 32),
                stride=data_cfg.get("stride", 16),
                batch_size=data_cfg.get("batch_size", 4),
                num_workers=data_cfg.get("num_workers", 0),
                max_episodes=data_cfg.get("max_episodes"),
                augment_brightness=data_cfg.get("augment_brightness", 0.0),
                augment_contrast=data_cfg.get("augment_contrast", 0.0),
                seed=data_cfg.get("seed"),
            ),
            optimizer=OptimizerConfig(
                lr=data.get("optimizer", {}).get("lr", 1e-4),
                weight_decay=data.get("optimizer", {}).get("weight_decay", 0.0),
            ),
            scheduler=SchedulerConfig(
                name=data.get("scheduler", {}).get("name", "none"),
                warmup_steps=data.get("scheduler", {}).get("warmup_steps", 0),
            ),
            checkpoint=CheckpointConfig(
                dir=Path(data.get("checkpoint", {}).get("dir", "artifacts/checkpoints")),
                save_every=data.get("checkpoint", {}).get("save_every", 1000),
                keep_last=data.get("checkpoint", {}).get("keep_last", 2),
                resume=Path(data.get("checkpoint", {}).get("resume"))
                if data.get("checkpoint", {}).get("resume")
                else None,
            ),
            precision=data.get("precision", "bf16"),
            grad_clip_norm=data.get("grad_clip_norm", 1.0),
            grad_checkpointing=data.get("grad_checkpointing", True),
            compile_model=data.get("compile_model", False),
            seed=data.get("seed", None),
            mock_model=data.get("mock_model", False),
            wandb=WandbConfig(**data["wandb"]) if data.get("wandb") else None,
        )
        # Apply selective CLI overrides when a config file is provided.
        if args.total_steps is not None and args.total_steps != 1000:
            cfg.total_steps = args.total_steps
        if args.mock_model:
            cfg.mock_model = True
        if args.precision is not None and args.precision != cfg.precision:
            cfg.precision = args.precision
        if args.wandb_project:
            cfg.wandb = WandbConfig(
                project=args.wandb_project,
                entity=args.wandb_entity,
                mode=args.wandb_mode,
                run_name=args.wandb_run_name,
                tags=args.wandb_tag,
                group=args.wandb_group,
                log_artifacts=args.wandb_log_artifacts,
            )
        return cfg

    if not (args.data_root and args.spec and args.norms):
        raise SystemExit("Must supply --config or --data-root/--spec/--norms.")

    return TrainingConfig(
        total_steps=args.total_steps,
        data=DataConfig(
            data_root=args.data_root,
            spec_path=args.spec,
            norms_path=args.norms,
            seq_len=args.seq_len,
            stride=args.stride,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
        ),
        optimizer=OptimizerConfig(lr=args.lr, weight_decay=args.weight_decay),
        checkpoint=CheckpointConfig(
            dir=args.checkpoint_dir,
            save_every=args.save_every,
            resume=args.resume,
        ),
        precision=args.precision,
        grad_checkpointing=not args.no_grad_checkpointing,
        compile_model=args.compile_model,
        seed=args.seed,
        mock_model=args.mock_model,
        wandb=WandbConfig(
            project=args.wandb_project,
            entity=args.wandb_entity,
            mode=args.wandb_mode,
            run_name=args.wandb_run_name,
            tags=args.wandb_tag,
            group=args.wandb_group,
            log_artifacts=args.wandb_log_artifacts,
        )
        if args.wandb_project
        else None,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    import logging

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = parse_config(args)
    runner = TrainingRunner(config)
    runner.train()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

