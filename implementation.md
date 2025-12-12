0) Repo scaffold (before coding features)
Decide stack: Python 3.10/3.11, PyTorch (ROCm), LeRobot 0.4.2 pinned.
Create structure: configs/, data/ (ignored), scripts/, src/ (pkg), tests/, artifacts/ (ignored), ci/.
Tooling: poetry/pip-tools or uv for lock; ruff + black + mypy (strict where feasible); pre-commit; pytest.
Config system: OmegaConf/Hydra or pydantic config files in configs/. Favor declarative YAML + schema validation.
Logging: loguru/structlog + W&B; env-based log level.

1) Pin and verify environment (ROCm-friendly)
Lock dependencies: torch/rocm wheels, LeRobot 0.4.2, torchvision, wandb, numpy, scipy, opencv, albumentations/kornia, rich, hydra/omegaconf, einops, tqdm.
Add requirements.lock and a scripts/bootstrap_rocm.sh with ROCm checks (GPU visibility, drivers).
Add a scripts/check_env.py to print torch, ROCm, GPUs, LeRobot version.

2) Dataset contract + validation (fail fast)
Define dataset schema (JSON/YAML): modalities (camera keys, proprio keys), action spec (DoF count, joint order, units, delta/absolute), control freq target and tolerance, max episode length.
Implement src/data/validator.py:
Episode boundary consistency; len(obs)==len(actions)
Control frequency tolerance check
Required keys present; shapes consistent
NaN/Inf checks
Action range sanity (per-joint min/max)
Corrupt frame handling policy (fail or drop with report)
CLI: python -m src.cli.validate --data ... --spec ... --report ...
Output: structured report (JSON) + human-readable summary.

3) Normalization artifacts (compute once, freeze)
Implement src/data/norms.py:
Compute per-dim mean/std for actions (+ proprio if used) on train split only.
Save artifact (e.g., artifacts/norms/{dataset_hash}/norms.npz + metadata.json with dataset hash, spec version, timestamp).
Log to W&B as immutable artifact.
CLI: python -m src.cli.compute_norms --data ... --spec ... --out ...

4) Preprocessing / transforms
Implement src/data/transforms.py:
Apply frozen normalization; conservative image augs (brightness/contrast, optional blur) configurable; deterministic seeding across timesteps.
Ensure temporal alignment preserved; no shuffling inside sequences.
Config: enable/disable per modality; default bounds documented.

5) Datasets & loaders
Implement src/data/dataset.py to wrap LeRobotDataset with:
Action spec assertions (DoF/order/semantics)
Split handling (train/val)
Sequence batching (context len for Pi05)
Collate with padding/masking if needed
Dataloader seeding and worker_init_fn for determinism.

6) Training orchestration (Pi05 fine-tune)
Implement src/train/runner.py:
Load config, verify dataset + norms presence, log resolved config + git hash + env.
Instantiate Pi05 policy from LeRobot 0.4.2.
Controls: total steps, batch/seq len, LR & scheduler, precision (bf16/fp16/fp32), grad checkpointing, torch.compile toggle.
Checkpointing: schema (model, optimizer, scheduler, scaler, step, config, norms ref). Save best + last; cadence configurable.
Resume: load everything including optimizer/scheduler; validate config compatibility.
CLI: python -m src.cli.train --config configs/phase_a.yaml --resume optional.

7) Two-phase defaults (configs)
configs/phase_a.yaml: sanity finetune (3k–6k steps, small batch, frequent checkpoints).
configs/phase_b.yaml: scale finetune (20k–50k+), adjusted LR/scheduler, checkpoint every N steps.
Include ROCm defaults: compile off, grad checkpointing on, prefer bf16 if stable else fp16.

8) Evaluation pipeline (offline)
Implement src/eval/offline.py:
Metrics: action MSE (norm/unnorm), per-joint stats, Δ-magnitude smoothness, saturation/clipping counts.
Logging: predicted vs GT traces, representative episode frame strips/videos.
CLI: python -m src.cli.eval --checkpoint ... --data ... --out ...
Decide which checkpoints to eval (best/last/every N).

9) Experiment tracking (W&B)
Standardize run metadata: resolved config, git commit, LeRobot version, torch/ROCm/GPU info.
Artifacts: dataset ref + hash, norms, checkpoints.
Media: curves, action traces, eval videos.
Add a lightweight wrapper src/utils/wandb.py to enforce fields and artifact names.

10) Reproducibility & safety rails
Global seeding util (torch, numpy, random, dataloader workers); deterministic flags where possible on ROCm.
Log env snapshot: pip freeze, ROCm/driver, GPU model, LeRobot hash.
Guardrails: assert norms exist before train; assert action spec matches; refuse mixed dataset versions unless explicitly allowed.

11) CI / smoke
GitHub Actions (if available) or local scripts/smoke.sh:
python -m src.cli.validate --data dummy
python -m src.cli.compute_norms --data dummy
Tiny train (few steps, CPU/rocm) to catch wiring errors.
python -m src.cli.eval on tiny checkpoint.

12) Docs and checklists
README: quickstart, environment setup (ROCm), commands for validate/norm/train/eval.
docs/ for dataset spec schema, action contract, artifact schemas, config fields.
Checklists for “ready to train” and “ready to deploy”.
If you want, I can draft the initial repo layout (directories, placeholder files, sample configs) and the first set of CLIs; just switch to agent mode and I’ll scaffold it.