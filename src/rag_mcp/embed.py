"""Embedding client for the llama.cpp router (OpenAI /v1/embeddings API).
qwen3-embedding is instruction-aware: queries get an instruct prefix, documents
are embedded raw."""

import httpx

from .config import Config

QUERY_PREFIX = (
    "Instruct: Given a question about past coding/agent sessions, retrieve relevant "
    "session notes\nQuery: "
)

_BATCH = 32


async def embed_texts(cfg: Config, texts: list[str], *, is_query: bool = False) -> list[list[float]]:
    if not texts:
        return []
    inputs = [QUERY_PREFIX + t if is_query else t for t in texts]
    out: list[list[float]] = []
    async with httpx.AsyncClient(timeout=60) as client:
        for i in range(0, len(inputs), _BATCH):
            batch = inputs[i : i + _BATCH]
            for attempt in (1, 2):
                try:
                    r = await client.post(
                        f"{cfg.embed_url}/embeddings",
                        json={"model": cfg.embed_model, "input": batch},
                    )
                    r.raise_for_status()
                    data = r.json()["data"]
                    out.extend(d["embedding"] for d in sorted(data, key=lambda d: d["index"]))
                    break
                except Exception:
                    if attempt == 2:
                        raise
    return out


async def embed_query(cfg: Config, text: str) -> list[float]:
    return (await embed_texts(cfg, [text], is_query=True))[0]
