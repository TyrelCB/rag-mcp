# rag-mcp

Hybrid RAG over Claude Code and Hermes session history on this box, plus an SFT
export + LoRA fine-tuning pipeline for distilling frontier-model sessions into
small local models.

One persistent service (port **8004**, systemd user unit `rag-mcp`) provides:

- **MCP** (streamable HTTP, `https://rag.mcp.tyrel.cloud/mcp`): `rag_search`,
  `rag_ingest_text`, `rag_status` — registered in both Claude Code
  (`~/.claude.json`) and Hermes (`~/.hermes/config.yaml` → `rag-mcp`).
- **REST for hooks**:
  - `POST /api/ingest` — enqueue a session transcript (202, background worker)
  - `POST /api/context` — hybrid search, returns a provenance-tagged context block
  - `GET /health` — store stats + queue depth

## How data flows

```
Claude Code SessionEnd  ─┐
Hermes on_session_end   ─┼─► POST /api/ingest ─► queue ─► parse ─► junk filter
                         │      (pending_jobs table survives restarts)
                         │   ─► distill (llama.cpp :9090, Qwen3.6-35b-1M) ─► chunk
                         │   ─► scrub secrets ─► embed (llama.cpp :9090, qwen3-embedding-0.6b)
                         │   ─► SQLite: chunks + FTS5 + sqlite-vec
Claude Code UserPromptSubmit ─► POST /api/context ─► vec KNN + BM25 → RRF → boosts
                                 → dedupe (per-session `injected` cache) → inject
```

Store: `~/.local/share/rag-mcp/rag.db` (WAL). Chunk kinds: `summary`, `fact`,
`error_fix`, `user_prompt`, `assistant_answer`, `code`, `manual`.

Hooks (all fail-open — service down means silence, never a blocked prompt):
- `~/.claude/hooks/claude-rag-context.sh` (UserPromptSubmit), `claude-rag-ingest.sh`
  (SessionEnd) — wired in `~/.claude/settings.json`.
- `~/.hermes/agent-hooks/hermes-rag-ingest.sh` — wired in `~/.hermes/config.yaml`
  `hooks:` block, allowlisted in `~/.hermes/shell-hooks-allowlist.json`.

## CLI

```bash
rag-mcp serve                     # what systemd runs
rag-mcp status
rag-mcp backfill --source all     # seed from existing history (--no-distill for speed)
rag-mcp ingest <path> --source claude
rag-mcp export --out data/sft-$(date +%Y%m%d) --min-turns 3
rag-mcp reembed --model <router-model-id> --dim <n>   # switch embedding models
```

## Fine-tuning (training/)

1. `rag-mcp export --out data/sft-YYYYMMDD` → `train_tools.jsonl` (full tool
   trajectories), `train_chat.jsonl` (text-only), val splits, `stats.json`.
   Quality gates: frontier (claude*) model, ≥N user turns, <30% tool errors, no
   failure endings, dedup; secrets redacted.
2. `training/run_container.sh python train_lora.py --base Qwen/Qwen3.5-9B \
   --data /ws/data/sft-YYYYMMDD/train_tools.jsonl --run-name my-run` — bf16 LoRA
   via TRL/PEFT inside the NGC pytorch container (aarch64; no bitsandbytes).
   Smoke: `--base Qwen/Qwen3-0.6B --max-steps 2`.
3. `training/merge_and_export.sh runs/my-run Qwen/Qwen3.5-9B tyrel-tuned-qwen` —
   merge → GGUF (`~/llama.cpp`) → Q4_K_M → preset in `~/models/presets.ini`,
   served by the llama.cpp router on :9090.

## Config (env)

`PORT` (8004) · `RAG_DB` · `RAG_EMBED_URL`/`RAG_EMBED_MODEL` (llama.cpp router
`/v1/embeddings`, qwen3-embedding-0.6b via `~/models/presets.ini`, dim recorded
in `meta`; mismatch refuses startup) · `RAG_DISTILL_URL`/`RAG_DISTILL_MODEL`
(llama.cpp :9090, Qwen3.6-35b-1M-P1-MTP-NGRAM) · `RAG_CONTEXT_TOKENS` (1500) ·
`RAG_MIN_SESSION_CHARS` (700).

## Dev

```bash
uv sync && .venv/bin/python -m pytest tests/ -q
```
