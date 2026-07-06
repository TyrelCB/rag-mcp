#!/usr/bin/env python3
"""Benchmark a tuned model against its base via the llama.cpp router.

Blind pairwise A/B: for each eval prompt, both models answer, then a judge
model picks a winner. Answer order is randomized per prompt to cancel the
judge's position bias. Runs on the host (stdlib + httpx only, uses the rag-mcp
venv): no GPU containers needed.

  .venv/bin/python training/eval_compare.py \
      --tuned tyrel-qwen35-9b-sessions --base <base-model-id-on-router> \
      --data data/sft-20260706/val_tools.jsonl [--judge Qwen3.6-35b-1M-P1-MTP-NGRAM] [--n 20]

For a scalar quality metric, also compare held-out loss: run train_lora.py
with --max-steps 1 --epochs 0 twice (once per model) and read eval_loss, or
use llama-perplexity on the GGUFs.
"""

import argparse
import json
import random
import re

import httpx

ROUTER = "http://127.0.0.1:9090/v1"

JUDGE_PROMPT = """You are grading two AI assistant responses to the same request.
Judge on: correctness, specificity to the user's environment, actionability, and honesty
(admitting unknowns beats confident nonsense). Ignore length and formatting style.

REQUEST:
{prompt}

RESPONSE A:
{a}

RESPONSE B:
{b}

Reply with exactly one line: "WINNER: A", "WINNER: B", or "WINNER: TIE", then one sentence why."""


def chat(client: httpx.Client, model: str, messages: list, max_tokens: int = 600) -> str:
    r = client.post(f"{ROUTER}/chat/completions", json={
        "model": model, "messages": messages,
        "max_tokens": max_tokens, "temperature": 0.2,
    })
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"] or ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tuned", required=True, help="tuned model id on the router")
    ap.add_argument("--base", required=True, help="base model id on the router")
    ap.add_argument("--judge", default="Qwen3.6-35b-1M-P1-MTP-NGRAM")
    ap.add_argument("--data", required=True, help="SFT JSONL; first user msg of each row becomes a prompt")
    ap.add_argument("--n", type=int, default=20, help="max prompts")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    prompts = []
    for line in open(args.data):
        msgs = json.loads(line)["messages"]
        user = next((m["content"] for m in msgs if m["role"] == "user"), None)
        if user and len(user) > 30:
            prompts.append(user[:4000])
    random.Random(args.seed).shuffle(prompts)
    prompts = prompts[: args.n]
    print(f"{len(prompts)} prompts · tuned={args.tuned} vs base={args.base} · judge={args.judge}\n")

    system = [{"role": "system", "content": "You are a capable software engineering assistant."}]
    wins = {"tuned": 0, "base": 0, "tie": 0, "error": 0}
    rng = random.Random(args.seed)

    with httpx.Client(timeout=600) as client:
        for i, p in enumerate(prompts, 1):
            try:
                tuned_out = chat(client, args.tuned, system + [{"role": "user", "content": p}])
                base_out = chat(client, args.base, system + [{"role": "user", "content": p}])
                tuned_is_a = rng.random() < 0.5
                a, b = (tuned_out, base_out) if tuned_is_a else (base_out, tuned_out)
                verdict = chat(client, args.judge, [{
                    "role": "user",
                    "content": JUDGE_PROMPT.format(prompt=p[:2000], a=a[:3000], b=b[:3000]),
                }], max_tokens=120)
                m = re.search(r"WINNER:\s*(A|B|TIE)", verdict, re.I)
                pick = (m.group(1).upper() if m else "TIE")
                if pick == "TIE":
                    wins["tie"] += 1
                elif (pick == "A") == tuned_is_a:
                    wins["tuned"] += 1
                else:
                    wins["base"] += 1
                print(f"[{i}/{len(prompts)}] {pick if pick=='TIE' else ('tuned' if (pick=='A')==tuned_is_a else 'base')} — {p[:70]!r}")
            except Exception as e:
                wins["error"] += 1
                print(f"[{i}/{len(prompts)}] error: {e}")

    total = wins["tuned"] + wins["base"] + wins["tie"]
    print(f"\ntuned wins: {wins['tuned']}  base wins: {wins['base']}  ties: {wins['tie']}  errors: {wins['error']}")
    if total:
        print(f"tuned win rate (excl. ties): "
              f"{wins['tuned'] / max(1, wins['tuned'] + wins['base']):.0%}")


if __name__ == "__main__":
    main()
