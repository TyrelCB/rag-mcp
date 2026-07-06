#!/usr/bin/env bash
# Run a training command inside the NGC pytorch container (aarch64 + Blackwell).
# Usage: ./run_container.sh python train_lora.py --base Qwen/Qwen3-0.6B --data ... --max-steps 2
set -euo pipefail

IMAGE="${NGC_IMAGE:-nvcr.io/nvidia/pytorch:25.12-py3}"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"

mkdir -p "$HF_CACHE"

TTY_FLAGS=""
[ -t 0 ] && TTY_FLAGS="-it"

docker run --rm $TTY_FLAGS \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$PROJECT_DIR:/ws" \
  -v "$HF_CACHE:/root/.cache/huggingface" \
  -w /ws/training \
  "$IMAGE" \
  bash -c "pip uninstall -q -y torchao 2>/dev/null; pip install -q 'trl>=0.15' 'peft>=0.14' 'datasets>=3' && $* ; status=\$?; chown -R $(id -u):$(id -g) /ws/training/runs 2>/dev/null; exit \$status"
# torchao is uninstalled because the NGC image ships a 0.15 git build that PEFT's
# LoRA dispatcher rejects; plain bf16 LoRA doesn't need it. The container runs as
# root (system site-packages must be writable), so outputs get chowned back.
