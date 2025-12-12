Pi05 + LeRobot (0.4.2) Training Repo Specification

This document defines what must be implemented and why to build a production-grade training repository for fine-tuning π₀.₅ (Pi05) with LeRobot v0.4.2, targeting arm-only control on LeKiwi, running on AMD / ROCm GPUs.

This is a conceptual + procedural spec. It intentionally avoids directory layout and code-level details. The goal is to give an ML engineer a clear, unambiguous contract for what the training repo guarantees.

⸻

0. Scope & assumptions

In scope
	•	Offline fine-tuning of Pi05 using LeRobotDataset provided by another engineer
	•	Arm-only actions (no base control)
	•	AMD cloud GPU (ROCm)
	•	Experiment tracking (Weights & Biases)
	•	Reproducible, resumable training

Out of scope
	•	Model architecture changes to Pi05
	•	Custom loss design
	•	Online RL or on-robot learning

Hard constraints
	•	LeRobot version is locked to 0.4.2
	•	Pi05 policy implementation comes from LeRobot (not re-implemented)

⸻

1. Core design principles (why this repo exists)
	1.	Separation of concerns
The repo owns training correctness, not robotics plumbing or model internals.
	2.	Fail fast on bad data
Most imitation failures come from silent data issues. Validation is mandatory.
	3.	Normalization is a first-class artifact
Statistics used at training time must be versioned, frozen, and reproducible.
	4.	Config > code
Training behavior must be controllable without editing Python files.
	5.	Observability by default
If it’s not logged, it didn’t happen.

⸻

2. Step 1 — Dataset ingestion & validation

What must be implemented

A dataset validation step that runs before any training or normalization.

Required checks

The validator must explicitly verify:
	•	Episode boundaries are present and consistent
	•	Observation/action sequence lengths match
	•	Fixed control frequency (within tolerance)
	•	Required observation keys exist (camera(s), optional proprio)
	•	Action tensor shape is consistent across episodes
	•	No NaNs / Infs in observations or actions
	•	Action values fall within sane physical limits

Why this matters
	•	Pi05 learns temporal causality. Misalignment silently destroys learning.
	•	Bad episodes poison the loss but look “numerically fine”.
	•	Catching issues here saves GPU days later.

Definition of done: validation either exits cleanly with a report, or hard-fails with actionable errors.

⸻

3. Step 2 — Define the action contract (arm-only)

What must be implemented

A single, explicit action specification that applies to the entire dataset and training run.

This includes:
	•	Number of arm DoFs
	•	Ordering of joints
	•	Action semantics (e.g. joint deltas vs joint targets)
	•	Units and expected ranges

Why this matters
	•	Pi05 treats actions as continuous tokens; semantic ambiguity breaks imitation
	•	Mixing conventions inside one dataset guarantees failure
	•	This contract becomes part of the model’s implicit interface

Definition of done: one written action spec + runtime assertion that dataset matches it.

⸻

4. Step 3 — Compute and freeze normalization statistics

What must be implemented

A one-time preprocessing step that computes normalization statistics from the training split only.

At minimum:
	•	Per-dimension mean and std (or equivalent) for actions
	•	Same for proprioception if used

Statistics must be:
	•	Saved as a standalone artifact (JSON / NPZ / etc.)
	•	Hash-linked to the dataset version
	•	Logged to W&B as an immutable artifact

Why this matters
	•	Pi-style VLA models are extremely sensitive to action scaling
	•	Inconsistent scales bias gradients toward large-magnitude joints
	•	Recomputing stats between runs makes results non-comparable

Definition of done: training cannot start unless normalization artifacts exist and are loaded.

⸻

5. Step 4 — Dataset preprocessing & transforms

What must be implemented

A preprocessing layer that:
	•	Applies frozen normalization to actions (and proprio)
	•	Applies light, configurable image augmentations
	•	Preserves temporal alignment exactly

Augmentations should be:
	•	Conservative by default (brightness/contrast jitter)
	•	Entirely config-controlled

Why this matters
	•	Normalization is part of the model, not a data detail
	•	Over-aggressive augmentation breaks manipulation
	•	Configurable transforms allow rapid ablation studies

Definition of done: same raw dataset + same config → identical tensors.

⸻

6. Step 5 — Training loop orchestration (Pi05 via LeRobot)

What must be implemented

A training entrypoint that:
	•	Loads configs
	•	Verifies dataset + normalization artifacts
	•	Launches Pi05 fine-tuning using LeRobot’s policy implementation
	•	Supports resume from checkpoint

Mandatory training controls

Expose the following as config (not hardcoded):
	•	Total training steps
	•	Batch size / sequence length
	•	Learning rate & scheduler
	•	Precision mode (bf16 / fp16 / fp32)
	•	Gradient checkpointing
	•	torch.compile on/off

AMD / ROCm defaults (recommended)
	•	Start with compile_model = false
	•	Enable gradient checkpointing
	•	Prefer bf16 if stable, otherwise fp16

Why this matters
	•	LeRobot already encodes Pi05 best practices — reuse them
	•	AMD stacks are less forgiving to dynamic graph changes
	•	Config-driven control prevents accidental regressions

Definition of done: a single command launches a resumable training run.

⸻

7. Step 6 — Training step budgeting & finetuning strategy

What must be implemented

A documented, enforced two-phase training strategy:

Phase A — Sanity finetune
	•	3k–6k steps
	•	Small batch
	•	Frequent checkpoints

Phase B — Scale finetune
	•	20k–50k+ steps (if justified)
	•	Adjust LR / scheduler

Why this matters
	•	Early detection of data or config issues
	•	Avoids burning GPU time on broken setups
	•	Matches best practice for imitation learning

Definition of done: configs clearly separate sanity vs scale runs.

⸻

8. Step 7 — Evaluation (offline, mandatory)

What must be implemented

An offline evaluation pipeline that runs without a robot and computes:
	•	Action MSE (normalized & unnormalized)
	•	Per-joint error statistics
	•	Action smoothness metrics (Δ-magnitude)
	•	Saturation / clipping counts

Additionally:
	•	Log predicted vs ground-truth action traces
	•	Log representative episode videos or frame sequences

Why this matters
	•	Loss alone does not predict behavior quality
	•	Offline diagnostics catch frozen or unstable policies
	•	Makes model selection objective

Definition of done: every checkpoint can be evaluated with the same protocol.

⸻

9. Step 8 — Experiment tracking (Weights & Biases)

What must be implemented

W&B must log, at minimum:

Run metadata
	•	Full resolved config
	•	Git commit hash
	•	LeRobot version (0.4.2)
	•	Torch / ROCm / GPU info

Artifacts
	•	Dataset reference + hash
	•	Normalization stats
	•	Checkpoints (best + last)

Media
	•	Training curves
	•	Action trace plots
	•	Offline eval videos

Why this matters
	•	Enables comparison across datasets, seeds, configs
	•	Allows clean handoff between engineers
	•	Essential for debugging regression

Definition of done: a run can be fully reconstructed from W&B alone.

⸻

10. Step 9 — Reproducibility & safety rails

What must be implemented
	•	Global seeding (torch, numpy, dataloaders)
	•	Deterministic flags where possible
	•	Explicit environment/version logging

Optional but recommended:
	•	CI smoke test (dataset validation + tiny train)

Why this matters
	•	Robotics datasets are expensive to collect
	•	Non-reproducible training wastes data and trust

Definition of done: two runs with same inputs produce statistically equivalent outputs.

⸻

11. Step 10 — Handoff contract to the ML engineer

The final repo should guarantee:
	•	“If the dataset passes validation, training will not silently fail.”
	•	“Normalization is consistent across all runs.”
	•	“Every experiment is logged, resumable, and comparable.”
	•	“Pi05 behavior differences are attributable to data or config — not plumbing.”

This is the definition of a professional-grade training environment for Pi05 + LeRobot.

⸻

Appendix — Mental model

Think of this repo as:

A compiler for robotics datasets → policies

Correctness, observability, and reproducibility matter more than clever tricks.

⸻

End of specification