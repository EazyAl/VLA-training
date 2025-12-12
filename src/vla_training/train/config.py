from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

Precision = Literal["bf16", "fp16", "fp32"]


@dataclass
class DataConfig:
    data_root: Path
    spec_path: Path
    norms_path: Path
    seq_len: int = 32
    stride: int = 16
    batch_size: int = 4
    num_workers: int = 0
    max_episodes: Optional[int] = None
    augment_brightness: float = 0.0
    augment_contrast: float = 0.0
    seed: Optional[int] = None


@dataclass
class OptimizerConfig:
    lr: float = 1e-4
    weight_decay: float = 0.0


@dataclass
class SchedulerConfig:
    name: Literal["none", "cosine"] = "none"
    warmup_steps: int = 0


@dataclass
class CheckpointConfig:
    dir: Path
    save_every: int = 1000
    keep_last: int = 2
    resume: Optional[Path] = None


@dataclass
class TrainingConfig:
    total_steps: int
    data: DataConfig
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    checkpoint: CheckpointConfig = field(default_factory=lambda: CheckpointConfig(dir=Path("artifacts/checkpoints")))
    precision: Precision = "bf16"
    grad_clip_norm: Optional[float] = 1.0
    grad_checkpointing: bool = True
    compile_model: bool = False
    seed: Optional[int] = 42
    mock_model: bool = False  # Use a tiny dummy model for quick CPU smoke tests
    wandb: Optional["WandbConfig"] = None


@dataclass
class WandbConfig:
    project: str
    entity: Optional[str] = None
    run_name: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    mode: Literal["online", "offline", "disabled"] = "disabled"
    log_artifacts: bool = False
    group: Optional[str] = None

