# Remote GPU Training Setup Guide

> **For Independent Remote Setup**: If you cannot connect from local to remote, use the automated setup scripts:
> - **Bash script**: `bash scripts/setup_remote.sh`
> - **Python script**: `python scripts/setup_remote.py`
> 
> These scripts will download the dataset from HuggingFace and set up everything automatically.

## What's Currently in the Repository

### ✅ **INCLUDED in Git (tracked):**
- **Source code**: All Python code in `src/vla_training/`
- **Configuration files**: `configs/phase_a.json`, `configs/phase_b.json`, `configs/specs/mechalabs.yaml`
- **Scripts**: `scripts/bootstrap_rocm.sh`, `scripts/check_env.py`, etc.
- **Dependencies**: `requirements.txt`, `requirements-dev.txt`, `pyproject.toml`
- **Tests**: All test files in `tests/`
- **Documentation**: README.md, implementation.md, training-plan.md

### ❌ **NOT INCLUDED in Git (ignored):**
- **Data directory** (`data/`): All dataset files are ignored by `.gitignore`
  - Currently you have `data/lekiwi_pickplace/` locally with episodes, but this is NOT in the repo
  - The repo only tracks the directory structure (via `.gitkeep` if present)
  
- **Artifacts directory** (`artifacts/`): All generated files are ignored
  - Normalization files: `artifacts/norms/lekiwi_pickplace/` (NOT in repo)
  - Model checkpoints: `artifacts/checkpoints/` (NOT in repo)
  - Validation reports: `artifacts/validate_*.json` (NOT in repo)

- **Virtual environments**: `venv/`, `.venv/` (NOT in repo)

## What You Need on the Remote GPU Machine

### 1. **Repository Code** ✅
Already there if you cloned the repo:
```bash
git clone https://github.com/EazyAl/VLA-training.git
cd VLA-training
```

### 2. **Dataset Files** ❌ (MUST TRANSFER)
The training data is NOT in the repo. You need to transfer:
```bash
# Option A: Use rsync/scp to copy data directory
rsync -avz data/lekiwi_pickplace/ user@remote:/path/to/VLA-training/data/lekiwi_pickplace/

# Option B: Use a shared filesystem/NFS mount
# Option C: Download from your data source on the remote machine
```

### 3. **Normalization Artifacts** ❌ (MUST TRANSFER or RECOMPUTE)
The computed normalization stats are NOT in the repo:
```bash
# Option A: Transfer existing norms
rsync -avz artifacts/norms/ user@remote:/path/to/VLA-training/artifacts/norms/

# Option B: Recompute on remote (recommended if data is already there)
python -m vla_training.cli.compute_norms \
  --data-root data/lekiwi_pickplace \
  --spec configs/specs/mechalabs.yaml \
  --out-dir artifacts/norms/lekiwi_pickplace \
  --max-episodes 50
```

### 4. **Model Checkpoints** ❌ (NOT NEEDED for fresh training)
- Checkpoints are saved during training to `artifacts/checkpoints/`
- Only needed if resuming training (use `--resume` flag)
- Can be transferred later if needed

## Complete Remote Setup Checklist

### Option A: Dataset & Model Caching (Environment Already Setup) ⚡

If you already have the environment set up (PyTorch, ROCm, requirements installed):

```bash
# On remote GPU machine
git clone https://github.com/EazyAl/VLA-training.git
cd VLA-training

# Download dataset and cache models from HuggingFace
python scripts/prepare_data_and_cache.py

# Or with custom options
python scripts/prepare_data_and_cache.py \
    --data-dir data/lekiwi_pickplace \
    --max-episodes 100 \
    --skip-models  # Skip model caching if not needed
```

This script will:
- Download dataset from HuggingFace (`PRFitz/lekiwi-dataset-pick-place-red123`)
- Prepare episodes in `.npz` format
- Cache models that LeRobot/transformers might use (SigLIP, etc.)

### Option B: Full Automated Setup (Including Environment)

If you need to set up the environment too:

```bash
# On remote GPU machine
git clone https://github.com/EazyAl/VLA-training.git
cd VLA-training

# Run automated setup script (downloads data from HuggingFace)
bash scripts/setup_remote.sh

# Or use Python version
python scripts/setup_remote.py
```

The script will:
1. Create virtual environment
2. Install ROCm PyTorch and dependencies
3. Download dataset from HuggingFace (`PRFitz/lekiwi-dataset-pick-place-red123`)
4. Prepare episodes in `.npz` format
5. Validate dataset
6. Compute normalization statistics

**Customization options:**
```bash
# Adjust number of episodes (default: 50)
MAX_EPISODES=100 bash scripts/setup_remote.sh

# Or with Python script
python scripts/setup_remote.py --max-episodes 100

# Skip dataset download if already exists
python scripts/setup_remote.py --skip-download

# Skip normalization if already computed
python scripts/setup_remote.py --skip-norms
```

### Option B: Manual Setup

### Step 1: Clone and Setup Environment
```bash
# On remote GPU machine
git clone https://github.com/EazyAl/VLA-training.git
cd VLA-training

# Create virtual environment
python3.10 -m venv venv  # or python3.11
source venv/bin/activate

# Install ROCm PyTorch (adjust version for your ROCm)
export TORCH_ROCM_VERSION=rocm6.2
pip install --extra-index-url https://download.pytorch.org/whl/$TORCH_ROCM_VERSION \
  torch==2.7.0+$TORCH_ROCM_VERSION \
  torchvision==0.22.0+$TORCH_ROCM_VERSION \
  torchaudio==2.7.0+$TORCH_ROCM_VERSION

# Install project dependencies
pip install -r requirements.txt
pip install -e .
```

### Step 2: Verify GPU/ROCm Setup
```bash
# Check ROCm installation
rocm-smi

# Verify PyTorch can see GPU
python scripts/check_env.py
```

### Step 3: Download and Prepare Data

**For independent remote setup (no local connection):**

The dataset is downloaded from HuggingFace and prepared automatically:

```bash
# Download and prepare lekiwi dataset
python scripts/prepare_lekiwi.py \
  --out-dir data/lekiwi_pickplace \
  --max-episodes 50  # Adjust as needed

# This will:
# 1. Download parquet files from PRFitz/lekiwi-dataset-pick-place-red123
# 2. Convert to .npz episodes in data/lekiwi_pickplace/episodes/
# 3. Extract arm-only actions (6 dims) and joint positions
```

**Alternative: If you have data elsewhere:**
```bash
# Option A: Transfer from local machine (if you can connect)
rsync -avz --progress data/lekiwi_pickplace/ user@remote:/path/to/VLA-training/data/lekiwi_pickplace/

# Option B: Download from cloud storage
# (Mount S3/GCS or download manually)
```

### Step 4: Validate Dataset (Optional but Recommended)
```bash
python -m vla_training.cli.validate \
  --data-root data/lekiwi_pickplace \
  --spec configs/specs/mechalabs.yaml \
  --report artifacts/validate_lekiwi.json \
  --max-episodes 10
```

### Step 5: Compute Normalization Stats
```bash
python -m vla_training.cli.compute_norms \
  --data-root data/lekiwi_pickplace \
  --spec configs/specs/mechalabs.yaml \
  --out-dir artifacts/norms/lekiwi_pickplace \
  --max-episodes 50 \
  --proprio-keys joint_pos joint_vel  # adjust based on your spec
```

### Step 6: Start Training
```bash
# Phase A (small sanity run)
python -m vla_training.cli.train --config configs/phase_a.json

# Phase B (full training run)
python -m vla_training.cli.train --config configs/phase_b.json

# Or with W&B enabled (edit phase_b.json to set wandb.mode: "online")
python -m vla_training.cli.train --config configs/phase_b.json --wandb-project vla-training
```

## Data Download Strategies (Independent Remote Setup)

### Strategy 1: HuggingFace Download (Current Setup) ✅
The `prepare_lekiwi.py` script automatically downloads from HuggingFace:
- Repository: `PRFitz/lekiwi-dataset-pick-place-red123`
- Downloads parquet chunks (no videos)
- Converts to `.npz` episodes locally
- No manual transfer needed

```bash
python scripts/prepare_lekiwi.py --out-dir data/lekiwi_pickplace --max-episodes 50
```

### Strategy 2: Direct HuggingFace Download
```bash
# Using huggingface-cli
pip install huggingface_hub[cli]
huggingface-cli download PRFitz/lekiwi-dataset-pick-place-red123 --local-dir data/lekiwi_pickplace
```

### Strategy 3: Cloud Storage Download
If your data is in cloud storage:
```bash
# AWS S3
aws s3 sync s3://your-bucket/data/ data/lekiwi_pickplace/

# Google Cloud Storage
gsutil -m cp -r gs://your-bucket/data/* data/lekiwi_pickplace/

# Azure Blob
az storage blob download-batch --source container --destination data/lekiwi_pickplace/
```

### Strategy 4: Shared Filesystem
- Mount NFS/network drive on remote machine
- Point `data_root` in configs to shared location
- No download needed if data is accessible

### Strategy 5: Manual Transfer (if you have connection)
```bash
# From local to remote (if you can connect)
rsync -avz --progress data/ user@remote:/path/to/VLA-training/data/
```

## Model Checkpoint Management

### During Training
- Checkpoints saved to: `artifacts/checkpoints/phase_a/` or `phase_b/`
- Format: `step_{N}.pt` (e.g., `step_1000.pt`)
- Contains: model state, optimizer, scheduler, scaler, step, config, spec

### Transferring Checkpoints
```bash
# Download checkpoint from remote
rsync -avz user@remote:/path/to/VLA-training/artifacts/checkpoints/phase_b/step_10000.pt ./

# Or sync entire checkpoint directory
rsync -avz user@remote:/path/to/VLA-training/artifacts/checkpoints/ ./artifacts/checkpoints/
```

### Resuming Training
Edit config file to add resume path:
```json
{
  "checkpoint": {
    "dir": "artifacts/checkpoints/phase_b",
    "save_every": 1000,
    "keep_last": 2,
    "resume": "artifacts/checkpoints/phase_b/step_10000.pt"
  }
}
```

## Monitoring Training on Remote Machine

### Option 1: W&B (Recommended)
```bash
# Enable in config or via CLI
python -m vla_training.cli.train --config configs/phase_b.json \
  --wandb-project vla-training \
  --wandb-mode online
```

### Option 2: SSH + tmux/screen
```bash
# Start training in tmux session
tmux new -s training
python -m vla_training.cli.train --config configs/phase_b.json
# Detach: Ctrl+B, then D
# Reattach: tmux attach -t training
```

### Option 3: Check Logs
Training logs to stdout. Redirect to file:
```bash
python -m vla_training.cli.train --config configs/phase_b.json 2>&1 | tee training.log
```

## Quick Verification Commands

```bash
# 1. Check environment
python scripts/check_env.py

# 2. Verify data exists
ls -lh data/lekiwi_pickplace/episodes/

# 3. Verify norms exist
ls -lh artifacts/norms/lekiwi_pickplace/

# 4. Test data loading (small batch)
python -c "
from vla_training.data import load_spec, EpisodeDataset, load_norms, Normalizer, SequenceConfig
spec = load_spec('configs/specs/mechalabs.yaml')
norms = load_norms('artifacts/norms/lekiwi_pickplace')
normalizer = Normalizer(
    action_mean=norms.action_mean,
    action_std=norms.action_std,
    proprio_mean=norms.proprio_mean,
    proprio_std=norms.proprio_std,
)
dataset = EpisodeDataset(
    spec=spec,
    data_root='data/lekiwi_pickplace',
    sequence=SequenceConfig(seq_len=32, stride=16),
    normalizer=normalizer,
    augment_fn=None,
    seed=123,
)
print(f'Dataset size: {len(dataset)}')
batch = dataset[0]
print(f'Batch keys: {batch.keys()}')
"
```

## Summary

| Item | In Repo? | Action Needed |
|------|----------|---------------|
| Source code | ✅ Yes | `git clone` |
| Config files | ✅ Yes | Already there |
| Dataset files | ❌ No | Transfer or download |
| Normalization stats | ❌ No | Transfer or recompute |
| Model checkpoints | ❌ No | Generated during training |
| Virtual environment | ❌ No | Create on remote |

**Critical**: You MUST transfer the `data/` directory and either transfer or recompute `artifacts/norms/` before training will work.

