# Training Guide for Remote GPU Machine

This guide walks you through training the Pi05 policy on your dataset on a remote GPU machine.

## Prerequisites

✅ Environment already set up (PyTorch, ROCm, requirements installed)  
✅ GPU accessible and verified

## Step-by-Step Training Process

### Step 1: Clone Repository and Setup

```bash
# On your remote GPU machine
git clone https://github.com/EazyAl/VLA-training.git
cd VLA-training

# IMPORTANT: Install the package in editable mode
# This is REQUIRED for the CLI commands to work
pip install -e .

# Verify installation
python -c "import vla_training; print('Package installed successfully')"
```

### Step 2: Download Dataset and Cache Models

```bash
# Download dataset from HuggingFace and cache Pi05 models
python scripts/prepare_data_and_cache.py

# This will:
# - Download dataset from PRFitz/lekiwi-dataset-pick-place-red123
# - Prepare episodes in .npz format
# - Cache Pi05 policy models (lerobot/pi05_base, lerobot/pi05_libero)
# - Cache underlying transformer models (SigLIP, Phi-1.5)
```

**Options:**
```bash
# Customize number of episodes
python scripts/prepare_data_and_cache.py --max-episodes 100

# Cache specific Pi05 model
python scripts/prepare_data_and_cache.py --pi05-model lerobot/pi05_libero

# Skip model caching if already done
python scripts/prepare_data_and_cache.py --skip-models
```

### Step 3: Validate Dataset (Optional but Recommended)

**Note**: Make sure you've run `pip install -e .` in Step 1, otherwise you'll get a `ModuleNotFoundError`.

```bash
# Validate the dataset to ensure it's correct
python -m vla_training.cli.validate \
    --data-root data/lekiwi_pickplace \
    --spec configs/specs/mechalabs.yaml \
    --report artifacts/validate_lekiwi.json \
    --max-episodes 10
```

### Step 4: Compute Normalization Statistics

**This is REQUIRED before training.** The normalization stats are computed from your dataset and used during training.

```bash
python -m vla_training.cli.compute_norms \
    --data-root data/lekiwi_pickplace \
    --spec configs/specs/mechalabs.yaml \
    --out-dir artifacts/norms/lekiwi_pickplace \
    --max-episodes 50 \
    --proprio-keys joint_pos
```

**What this does:**
- Computes mean/std for actions and proprioceptive keys
- Saves to `artifacts/norms/lekiwi_pickplace/norms.npz`
- This is a frozen artifact - use the same one for all training runs on this dataset

### Step 5: Start Training

#### Option A: Using Config File (Recommended)

```bash
# Phase A: Small sanity run (3000 steps, quick validation)
python -m vla_training.cli.train --config configs/phase_a.json

# Phase B: Full training run (20000 steps)
python -m vla_training.cli.train --config configs/phase_b.json
```

#### Option B: Using CLI Arguments

```bash
python -m vla_training.cli.train \
    --data-root data/lekiwi_pickplace \
    --spec configs/specs/mechalabs.yaml \
    --norms artifacts/norms/lekiwi_pickplace \
    --total-steps 3000 \
    --batch-size 2 \
    --seq-len 32 \
    --stride 16 \
    --lr 1e-4 \
    --checkpoint-dir artifacts/checkpoints/my_run \
    --save-every 500 \
    --precision bf16
```

### Step 6: Monitor Training

#### Check Logs
Training logs to stdout. Redirect to a file for later review:

```bash
python -m vla_training.cli.train --config configs/phase_a.json 2>&1 | tee training.log
```

#### Check Checkpoints
Checkpoints are saved to the directory specified in config:
```bash
ls -lh artifacts/checkpoints/phase_a/
# You'll see: step_500.pt, step_1000.pt, etc.
```

#### Use W&B (Optional)
Enable Weights & Biases for better monitoring:

```bash
# Edit configs/phase_b.json to set:
# "wandb": {
#   "project": "vla-training",
#   "mode": "online"
# }

# Or use CLI flags:
python -m vla_training.cli.train --config configs/phase_b.json \
    --wandb-project vla-training \
    --wandb-mode online
```

### Step 7: Resume Training (If Interrupted)

If training is interrupted, you can resume from a checkpoint:

```bash
# Edit configs/phase_b.json to add resume path:
# "checkpoint": {
#   "resume": "artifacts/checkpoints/phase_a/step_2000.pt"
# }

python -m vla_training.cli.train --config configs/phase_b.json
```

Or use CLI:
```bash
python -m vla_training.cli.train --config configs/phase_b.json \
    --resume artifacts/checkpoints/phase_a/step_2000.pt
```

## Training Configuration Files

### `configs/phase_a.json` - Sanity Run
- **Steps**: 3000
- **Batch size**: 2
- **Sequence length**: 32
- **Checkpoint every**: 500 steps
- **Purpose**: Quick validation that everything works

### `configs/phase_b.json` - Full Training
- **Steps**: 20000
- **Batch size**: 4
- **Sequence length**: 64
- **Checkpoint every**: 1000 steps
- **Scheduler**: Cosine with warmup
- **Purpose**: Full training run

## Key Training Parameters

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `total_steps` | Number of training steps | 3000-50000 |
| `batch_size` | Batch size | 2-8 (depends on GPU memory) |
| `seq_len` | Sequence length for temporal context | 32-128 |
| `stride` | Stride between sequences | 16-32 |
| `lr` | Learning rate | 1e-5 to 1e-4 |
| `precision` | Training precision | `bf16` (recommended for ROCm) |
| `grad_checkpointing` | Reduce memory usage | `true` (recommended) |
| `compile_model` | PyTorch 2.0 compilation | `false` (may not work on ROCm) |

## Troubleshooting

### Out of Memory (OOM)
- Reduce `batch_size`
- Reduce `seq_len`
- Enable `grad_checkpointing: true`
- Reduce `num_workers` to 0

### Slow Training
- Increase `batch_size` if memory allows
- Increase `num_workers` (if data loading is bottleneck)
- Check GPU utilization: `rocm-smi` or `nvidia-smi`

### ModuleNotFoundError: No module named 'vla_training'

**This means the package isn't installed.** Fix it with:

```bash
# Make sure you're in the repository root
cd /path/to/VLA-training

# Install the package in editable mode
pip install -e .

# Verify it works
python -c "import vla_training; print('OK')"
```

**Alternative**: If you can't install, you can run modules directly:
```bash
# Instead of: python -m vla_training.cli.validate
python src/vla_training/cli/validate.py --data-root ...
```

### Checkpoint Not Found
- Ensure normalization stats are computed first
- Check that dataset path is correct
- Verify spec file exists

### Model Download Issues
- Run `python scripts/prepare_data_and_cache.py` again
- Check HuggingFace cache: `~/.cache/huggingface/`
- Verify internet connection on remote machine

## Quick Reference Commands

```bash
# Complete setup (one-time)
git clone https://github.com/EazyAl/VLA-training.git
cd VLA-training

# CRITICAL: Install package first (required for all CLI commands)
pip install -e .

# Download dataset and cache models
python scripts/prepare_data_and_cache.py

# Compute normalization stats
python -m vla_training.cli.compute_norms \
    --data-root data/lekiwi_pickplace \
    --spec configs/specs/mechalabs.yaml \
    --out-dir artifacts/norms/lekiwi_pickplace

# Start training
python -m vla_training.cli.train --config configs/phase_a.json

# Resume training
python -m vla_training.cli.train --config configs/phase_b.json \
    --resume artifacts/checkpoints/phase_a/step_2000.pt

# Evaluate checkpoint
python -m vla_training.cli.eval \
    --checkpoint artifacts/checkpoints/phase_a/step_3000.pt \
    --data-root data/lekiwi_pickplace \
    --spec configs/specs/mechalabs.yaml \
    --norms artifacts/norms/lekiwi_pickplace
```

## Expected Output

During training, you'll see logs like:
```
[train] TrainingRunner init begin
[train] Building model...
[train] Model build complete.
Starting training for 3000 steps
step=1 loss=0.1234 lr=1.00e-04
step=2 loss=0.1123 lr=1.00e-04
...
Saved checkpoint at step 500 -> artifacts/checkpoints/phase_a/step_500.pt
```

## Next Steps After Training

1. **Evaluate the model**: Use `python -m vla_training.cli.eval` to test on validation episodes
2. **Compare checkpoints**: Evaluate different checkpoints to find the best one
3. **Deploy**: Use the best checkpoint for inference on your robot

## Notes

- **Pi05 models are cached**: After running `prepare_data_and_cache.py`, models won't re-download
- **Normalization is frozen**: Use the same norms file for all training runs on the same dataset
- **Checkpoints are incremental**: Each checkpoint saves model, optimizer, scheduler state
- **ROCm compatibility**: The code is tested with ROCm, but some features (like `compile_model`) may not work

