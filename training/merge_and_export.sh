#!/usr/bin/env bash
# Merge a LoRA adapter into its base model, convert to GGUF, quantize, and
# serve it from the llama.cpp router (:9090) via ~/models/presets.ini.
#
# Usage: ./merge_and_export.sh <run-dir> <base-hf-id> <router-model-name>
#   e.g. ./merge_and_export.sh runs/qwen35-9b-sessions-v1 Qwen/Qwen3.5-9B tyrel-qwen35-9b-sessions
#
# Runs the merge inside the NGC container (needs torch+peft); conversion and
# quantization use the host llama.cpp checkout. The router does NOT hot-reload
# presets.ini, so this restarts llama-server at the end (sudo).
#
# Fast path (no merge): convert the adapter alone and serve with llama-server:
#   python ~/llama.cpp/convert_lora_to_gguf.py <run-dir>/adapter --base <base-hf-id>
#   llama-server -m base.gguf --lora adapter.gguf
set -euo pipefail

RUN_REL="${1:?run dir (relative to training/)}"
BASE="${2:?base HF id}"
NAME="${3:?router model name}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUN_DIR="$SCRIPT_DIR/$RUN_REL"
LLAMA_CPP="${LLAMA_CPP:-$HOME/llama.cpp}"
PRESETS="${PRESETS:-$HOME/models/presets.ini}"
MERGED="$RUN_DIR/merged"
GGUF="$HOME/models/$NAME-q4_k_m.gguf"

if [ ! -d "$MERGED" ]; then
  "$SCRIPT_DIR/run_container.sh" python - <<EOF
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
base = AutoModelForCausalLM.from_pretrained("$BASE", dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base, "/ws/training/$RUN_REL/adapter")
model = model.merge_and_unload()
model.save_pretrained("/ws/training/$RUN_REL/merged")
AutoTokenizer.from_pretrained("$BASE").save_pretrained("/ws/training/$RUN_REL/merged")
print("merged")
EOF
fi

# --no-mtp: AutoModelForCausalLM drops Qwen3.5/3.6 MTP tensors at load, so the
# merged checkpoint has none; without the flag the GGUF declares an MTP block
# it can't fill and fails at serve time with "missing tensor blk.N.attn_norm".
python3 "$LLAMA_CPP/convert_hf_to_gguf.py" "$MERGED" --outtype bf16 --outfile "$RUN_DIR/model-bf16.gguf" --no-mtp
"$LLAMA_CPP/build/bin/llama-quantize" "$RUN_DIR/model-bf16.gguf" "$GGUF" Q4_K_M
rm -f "$RUN_DIR/model-bf16.gguf"

if ! grep -q "^\[$NAME\]" "$PRESETS"; then
  cat >> "$PRESETS" <<EOF

; Session-tuned model from rag-mcp training run $RUN_REL
[$NAME]
model = $GGUF
EOF
  echo "preset [$NAME] appended to $PRESETS"
fi

sudo systemctl restart llama-server
sleep 5
curl -s http://127.0.0.1:9090/v1/models | grep -q "$NAME" \
  && echo "done: $NAME is served by the router (:9090)" \
  || { echo "WARN: $NAME not visible on the router yet"; exit 1; }
