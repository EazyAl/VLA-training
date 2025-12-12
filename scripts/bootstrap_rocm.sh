#!/usr/bin/env bash
set -euo pipefail

echo "[bootstrap] Checking ROCm environment..."

if [[ ! -d "/opt/rocm" ]]; then
  echo "!! ROCm not found at /opt/rocm. Install ROCm before proceeding." >&2
  exit 1
fi

if command -v rocm-smi >/dev/null 2>&1; then
  echo "[bootstrap] rocm-smi detected. GPU info:"
  rocm-smi || true
else
  echo "!! rocm-smi not found in PATH. Ensure ROCm binaries are installed and PATH is set." >&2
fi

echo "[bootstrap] Recommended PyTorch (ROCm) install command:"
echo "  export TORCH_ROCM_VERSION=\"rocm6.2\"  # adjust to your ROCm install"
echo "  uv pip install --extra-index-url https://download.pytorch.org/whl/\${TORCH_ROCM_VERSION} \\"
echo "    torch==2.7.0+\${TORCH_ROCM_VERSION} torchvision==0.22.0+\${TORCH_ROCM_VERSION} torchaudio==2.7.0+\${TORCH_ROCM_VERSION}"

echo "[bootstrap] If using uv lock, include the extra index:"
echo "  uv lock --extra-index-url https://download.pytorch.org/whl/\${TORCH_ROCM_VERSION}"

echo "[bootstrap] Done. Verify with scripts/check_env.py after installation."

