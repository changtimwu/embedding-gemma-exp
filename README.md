# EmbeddingGemma 2 on MLX (local server)

OpenAI-compatible `/v1/embeddings` server for `mlx-community/embeddinggemma-2-bf16`
(text, image, audio, video → shared 768-d space), via mlx-vlm.

    uv run server.py                 # http://127.0.0.1:8080  (--host 0.0.0.0 for LAN)
    uv run test_client.py            # smoke test
    EMBED_MODEL=mlx-community/embeddinggemma-2-8bit uv run server.py   # smaller weights

## Request

    curl localhost:8080/v1/embeddings -H 'Content-Type: application/json' -d '{
      "input": ["Mars is the Red Planet.",
                [{"type": "text", "text": "A photo of"}, {"type": "image", "url": "/abs/path.jpg"}]],
      "prompt": "title: none | text: ",
      "dimensions": 256
    }'

- `input`: string, list of strings, or list items that are content-part lists
  (`text` / `image` / `audio` / `video` with `url` = local path or URL).
- `prompt` (optional): task prefix, e.g. `task: search result | query: ` for queries,
  `title: none | text: ` for documents. Or use `prompt_name` (`query`, `document`,
  `STS`, `Classification`, ... — full catalog at `GET /v1/models`); unknown names are a 400.
- `dimensions` (optional): 128/256/512/768 Matryoshka truncation, re-normalized.
  Use the same `dimensions` for queries and docs.

## Mixing images and text in one index

Image and text vectors share one space but occupy separate regions (the *modality
gap*): a matched image–caption pair scores lower than two paraphrased texts. Measured
on this server (BF16, M1 Max, 8 photos / 4 concepts):

| | CLIP ViT-B/32 | EmbeddingGemma 2 |
|---|---:|---:|
| image ↔ own caption | 0.32 | 0.73 |
| text ↔ paraphrase | 0.92 | 0.92 |
| image centroid ↔ text centroid | 0.28 | 0.83 |

The gap is far smaller than CLIP's but not zero, so in a mixed index a text query
favours text hits (correct image ranked 3/16, behind relevant texts). Task prefixes
don't fix this; z-scoring scores within each modality before merging does (correct
image 1/16 on all test queries). Treat cross-modal cosines as a ranking, not an
absolute threshold.

Why not `python -m mlx_vlm.server --embedding-model ...`? In mlx-vlm 0.7.8 its
embeddings route evaluates MLX on a different thread than it loaded on
(`RuntimeError: There is no Stream(gpu, 1)`), and `--embedding-model` doesn't become the
default model. `server.py` pins all MLX work to one thread.
