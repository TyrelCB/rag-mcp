#!/usr/bin/env bash
# Merge a LoRA adapter into its base model, convert to GGUF, quantize, and
# register with Ollama.
#
# Usage: ./merge_and_export.sh <run-dir> <base-hf-id> <ollama-name>
#   e.g. ./merge_and_export.sh runs/qwen95-sessions-v1 Qwen/Qwen3.5-9B tyrel-tuned-qwen3.5-9b
#
# Runs the merge inside the NGC container (needs torch+peft); conversion and
# quantization use the host llama.cpp checkout.
#
# Fast path (no merge): convert the adapter alone and serve with llama-server:
#   python ~/llama.cpp/convert_lora_to_gguf.py <run-dir>/adapter --base <base-hf-id>
#   llama-server -m base.gguf --lora adapter.gguf
set -euo pipefail

RUN_DIR="$(cd "$(dirname "$0")" && pwd)/${1:?run dir}"
BASE="${2:?base HF id}"
NAME="${3:?ollama model name}"
LLAMA_CPP="${LLAMA_CPP:-$HOME/llama.cpp}"
MERGED="$RUN_DIR/merged"

"$(dirname "$0")/run_container.sh" python - <<EOF
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
base = AutoModelForCausalLM.from_pretrained("$BASE", torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base, "/ws/training/${1}/adapter")
model = model.merge_and_unload()
model.save_pretrained("/ws/training/${1}/merged")
AutoTokenizer.from_pretrained("$BASE").save_pretrained("/ws/training/${1}/merged")
print("merged")
EOF

python3 "$LLAMA_CPP/convert_hf_to_gguf.py" "$MERGED" --outtype bf16 --outfile "$RUN_DIR/model-bf16.gguf"
"$LLAMA_CPP/build/bin/llama-quantize" "$RUN_DIR/model-bf16.gguf" "$RUN_DIR/model-q4_k_m.gguf" Q4_K_M

sed "s|{{GGUF_PATH}}|$RUN_DIR/model-q4_k_m.gguf|" "$(dirname "$0")/Modelfile.template" > "$RUN_DIR/Modelfile"
ollama create "$NAME" -f "$RUN_DIR/Modelfile"
echo "done: ollama run $NAME"
