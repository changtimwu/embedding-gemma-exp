"""OpenAI-compatible embedding server for EmbeddingGemma 2 on MLX.

POST /v1/embeddings
  {"input": "text" | ["text", ...] | [[{"type": "text", "text": ...},
                                        {"type": "image"|"audio"|"video", "url": ...}], ...],
   "dimensions": 128 | 256 | 512 | 768 (optional, Matryoshka truncation),
   "prompt": "task: search result | query: " (optional, prepended / system message),
   "prompt_name": "query" | "document" | ... (optional, from the model's prompt catalog)}

All MLX work runs on a single dedicated thread: MLX streams are thread-local.
"""

import argparse
import asyncio
import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Union

import mlx.core as mx
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from mlx_vlm import load
from mlx_vlm.utils import get_model_path

MODEL_ID = os.environ.get("EMBED_MODEL", "mlx-community/embeddinggemma-2-bf16")
DIMENSIONS = (128, 256, 512, 768)

executor = ThreadPoolExecutor(max_workers=1)
state: Dict[str, Any] = {}

Part = Dict[str, Any]


class EmbeddingsRequest(BaseModel):
    input: Union[str, List[Union[str, List[Part]]]]
    model: Optional[str] = None
    dimensions: Optional[int] = None
    prompt: Optional[str] = None
    prompt_name: Optional[str] = None
    encoding_format: Optional[str] = "float"


def _load():
    state["model"], state["processor"] = load(MODEL_ID)
    catalog = get_model_path(MODEL_ID) / "config_sentence_transformers.json"
    state["prompts"] = (
        json.loads(catalog.read_text()).get("prompts", {}) if catalog.exists() else {}
    )


def _embed_text(texts: List[str]) -> mx.array:
    inputs = state["processor"].tokenizer(texts, padding=True, return_tensors="np")
    out = state["model"](**{k: mx.array(v) for k, v in inputs.items()})
    return out.text_embeds, int(inputs["attention_mask"].sum())


def _embed_multimodal(items: List[List[Part]], prompt: Optional[str]) -> mx.array:
    conversations = []
    for parts in items:
        conv = [{"role": "system", "content": prompt}] if prompt else []
        conv.append({"role": "user", "content": parts})
        conversations.append(conv)
    inputs = state["processor"].apply_chat_template(
        conversations, tokenize=True, return_dict=True, return_tensors="mlx"
    )
    out = state["model"](**inputs)
    return out.text_embeds, int(inputs["attention_mask"].sum())


def _run(req: EmbeddingsRequest):
    items = [req.input] if isinstance(req.input, str) else req.input
    if not items:
        raise ValueError("`input` must be non-empty.")
    text_idx = [i for i, x in enumerate(items) if isinstance(x, str)]
    mm_idx = [i for i, x in enumerate(items) if not isinstance(x, str)]

    vectors: List[Optional[mx.array]] = [None] * len(items)
    tokens = 0
    if text_idx:
        prefix = req.prompt or ""
        emb, n = _embed_text([prefix + items[i] for i in text_idx])
        tokens += n
        for j, i in enumerate(text_idx):
            vectors[i] = emb[j]
    if mm_idx:
        emb, n = _embed_multimodal([items[i] for i in mm_idx], req.prompt)
        tokens += n
        for j, i in enumerate(mm_idx):
            vectors[i] = emb[j]

    emb = mx.stack(vectors)
    if req.dimensions and req.dimensions != emb.shape[-1]:
        emb = emb[:, : req.dimensions]
        emb = emb / mx.linalg.norm(emb, axis=-1, keepdims=True)
    mx.eval(emb)
    return emb.astype(mx.float32).tolist(), tokens


app = FastAPI(title="EmbeddingGemma 2 (MLX)")


@app.on_event("startup")
async def startup():
    executor.submit(_load).result()


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_ID}


@app.get("/v1/models")
async def models():
    return {
        "object": "list",
        "data": [{"id": MODEL_ID, "object": "model", "prompts": state["prompts"]}],
    }


@app.post("/v1/embeddings")
async def embeddings(req: EmbeddingsRequest):
    if req.dimensions is not None and req.dimensions not in DIMENSIONS:
        raise HTTPException(400, f"`dimensions` must be one of {DIMENSIONS}.")
    if req.prompt_name is not None:
        if req.prompt is not None:
            raise HTTPException(400, "Pass either `prompt` or `prompt_name`, not both.")
        if req.prompt_name not in state["prompts"]:
            raise HTTPException(
                400,
                f"Unknown `prompt_name`; expected one of {sorted(state['prompts'])}.",
            )
        req.prompt = state["prompts"][req.prompt_name]
    try:
        loop = asyncio.get_running_loop()
        vectors, tokens = await loop.run_in_executor(executor, _run, req)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        raise HTTPException(500, f"{type(exc).__name__}: {exc}")
    return {
        "object": "list",
        "data": [
            {"object": "embedding", "index": i, "embedding": v}
            for i, v in enumerate(vectors)
        ],
        "model": MODEL_ID,
        "usage": {"prompt_tokens": tokens, "total_tokens": tokens},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
