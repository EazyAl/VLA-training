# VLA Training (Pi05 + LeRobot 0.4.2)

This repository fine-tunes Pi05 with LeRobot v0.4.2 on AMD/ROCm for arm-only control. The plan and implementation steps live in `training-plan.md` and `implementation.md`.

## Current status
- Step 0 scaffold in place: directories, tooling config (ruff, black, mypy, pytest), pre-commit hooks planned, packaging via PEP 621 with hatchling.
- No runtime dependencies are added yet; they will be pinned in the environment step.

## Layout
- `src/` — package code (`vla_training`)
- `configs/` — configuration files (placeholder)
- `data/` — datasets (ignored; `.gitkeep` tracks the directory)
- `artifacts/` — generated artifacts (ignored)
- `scripts/` — utility scripts
- `tests/` — test suite
- `ci/` — CI-related scripts/configs

## Getting started
1. Create a Python 3.10/3.11 environment (ROCm-capable).
2. Install tooling via uv (recommended):
   - Install dev deps: `uv pip install -e .[dev]`
   - Lock (current platform): `uv lock --extra-index-url https://download.pytorch.org/whl/rocm6.2`
   - Install ROCm wheels (adjust rocm version if needed):
     `export TORCH_ROCM_VERSION=rocm6.2`
     `uv pip install --extra-index-url https://download.pytorch.org/whl/$TORCH_ROCM_VERSION torch==2.7.0+$TORCH_ROCM_VERSION torchvision==0.22.0+$TORCH_ROCM_VERSION torchaudio==2.7.0+$TORCH_ROCM_VERSION`
3. (Optional) Install pre-commit hooks: `pre-commit install`.
4. Verify environment: `python scripts/check_env.py`.

Subsequent steps will add dataset validation, normalization, training, and evaluation pipelines.

## Dataset validation (Step 2)
Expected layout (current validator assumption):
- `DATA_ROOT/episodes/*.npz`
- Each episode `.npz` contains:
  - `actions`: float array shaped (T, action_dim)
  - `timestamps`: float array shaped (T,) strictly increasing
  - observations: arrays named `obs.<key>` for each required observation/proprio key, each with first dimension T

Dataset spec (YAML/JSON) fields:
- `name`, `version`
- `control_freq_hz`, `control_freq_tolerance_pct`
- `required_obs_keys`, `required_proprio_keys`
- `action_dim`, optional `action_names`, `action_type` (delta|absolute)
- optional `action_min`/`action_max` (per-dim), `max_episode_steps`, `allow_nan`, `clip_actions`

Run validator:
```
python -m vla_training.cli.validate --data-root /path/to/data --spec /path/to/spec.yaml --report report.json
```
Use `--max-episodes` to cap validation for quick smoke passes.

## Normalization (Step 3)
- Computes per-dimension mean/std for actions and optional proprio keys over selected episodes.
- Output: `norms.npz` + `metadata.json` in a chosen directory.
- Run:
```
python -m vla_training.cli.compute_norms --data-root /path/to/data --spec /path/to/spec.yaml --out-dir artifacts/norms/run1 --max-episodes 50 --proprio-keys joint_pos joint_vel
```
If `--proprio-keys` is omitted, all `spec.required_proprio_keys` are used.

## Preprocessing & transforms (Step 4)
- `Normalizer`: applies frozen stats to `actions` and `obs.<proprio>` keys.
- `build_augmentations(AugmentConfig)`: optional brightness/contrast jitter for image observations (T, H, W, C), deterministic via provided RNG/seed.

## Datasets & loaders (Step 5)
- `EpisodeDataset`: loads `.npz` episodes, slices into windows (configurable `seq_len`/`stride`), applies optional augmentations and normalization.
- Expected episode keys remain: `actions`, `timestamps`, `obs.<key>`.
- Collation: `pad_and_collate` pads to the max length in batch and adds `attention_mask` and `lengths`.
- Build a loader:
```
from vla_training.data import (
    DatasetSpec, load_spec, EpisodeDataset, SequenceConfig,
    Normalizer, load_norms, build_augmentations, build_dataloader,
)
spec = load_spec("spec.yaml")
norms = load_norms("artifacts/norms/run1")
normalizer = Normalizer(
    action_mean=norms.action_mean,
    action_std=norms.action_std,
    proprio_mean=norms.proprio_mean,
    proprio_std=norms.proprio_std,
)
augment = build_augmentations(None)  # or AugmentConfig(...)
dataset = EpisodeDataset(
    spec=spec,
    data_root="/path/to/data",
    sequence=SequenceConfig(seq_len=32, stride=16),
    normalizer=normalizer,
    augment_fn=augment,
    seed=123,
)
loader = build_dataloader(dataset, batch_size=4, shuffle=True, num_workers=0, seed=42)
```

## Training orchestration (Step 6 - scaffold)
- Config dataclasses: `TrainingConfig` (with `DataConfig`, `OptimizerConfig`, `SchedulerConfig`, `CheckpointConfig`).
- Runner: `TrainingRunner` wires dataset/spec/norms, builds dataloader, model placeholder, optimizer, optional scheduler, checkpointing, and precision context.
- CLI: `python -m vla_training.cli.train --data-root ... --spec ... --norms ... --total-steps 1000 --checkpoint-dir artifacts/checkpoints`
- Note: the actual Pi05 model/loss is a placeholder (`NotImplementedError`) — integrate the LeRobot v0.4.2 Pi05 policy and loss where indicated in `train/runner.py`.
- Pi05 integration scaffold is now wired (PI05Policy from LeRobot 0.4.2). It infers input/output feature shapes from the dataset, builds dummy language tokens, and trains on the last timestep of each window. Replace with proper language/task handling when available.
- W&B support: pass `--wandb-project ... --wandb-mode online|offline` to enable logging; configs accept a `wandb` block.

## Phase configs (Step 7)
- `configs/phase_a.json`: sanity run (small batch/steps).
- `configs/phase_b.json`: longer run, cosine scheduler, W&B block (disabled by default).

## Offline eval (Step 8)
- CLI: `python -m vla_training.cli.eval --checkpoint path.pt --data-root ... --spec ... --norms ... --max-episodes 2 --mock-model`
- Reports MSE, per-dim MSE, delta magnitude, clipping fraction. `--mock-model` skips heavy model load.

## Experiment tracking (Step 9)
- W&B helper (`vla_training.utils.wandb`) and trainer hooks to log loss/lr per step when enabled.
- Enable via CLI flags or `wandb` block in configs (project/entity/mode/run_name/tags/group).

## Dataset fetch (MechaLabs placeholder)
- Script: `python scripts/download_mechalabs.py --out-dir data/mechalabs`
- The HF repo `CRPlab/MechaLabs` currently only lists README; script will warn and exit if empty.

## Phase A config example
- `configs/specs/mechalabs.yaml` — template spec (adjust to real dataset keys/action dims).
- `configs/phase_a.json` — sample training config pointing to `data/mechalabs` and `artifacts/norms/mechalabs`.

