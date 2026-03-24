"""
scripts/remote_embedder_server.py - 远程 Embedding 服务。

提供 OpenAI 兼容的 /v1/embeddings 接口，供本项目的 remote Embedder 调用。

启动示例：
    EMBEDDING_MODEL=BAAI/bge-m3 \
    EMBEDDING_DEVICE=cuda:0 \
    EMBEDDING_API_KEY=EMPTY \
    uvicorn scripts.remote_embedder_server:app --host 0.0.0.0 --port 8001
"""

from __future__ import annotations

import builtins
import logging
import os
from typing import Any, Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)5s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)

app = FastAPI(title="Remote Embedder", version="0.1.0")

_model: Any | None = None


class EmbeddingRequest(BaseModel):
    model: str
    input: str | list[str]


def _check_auth(authorization: str | None = Header(default=None)) -> None:
    expected = os.environ.get("EMBEDDING_API_KEY", "EMPTY")
    if expected in ("", "EMPTY"):
        return
    if authorization != f"Bearer {expected}":
        raise HTTPException(status_code=401, detail="Unauthorized")


def _patch_flagembedding_typing_compat() -> None:
    """
    某些 FlagEmbedding 版本的 BGE_M3/trainer.py 漏掉了 Optional 导入。
    在导入包前补到 builtins，可避免导入阶段直接 NameError。
    """
    if not hasattr(builtins, "Optional"):
        builtins.Optional = Optional


def _get_model():
    global _model
    if _model is not None:
        return _model

    _patch_flagembedding_typing_compat()

    try:
        from FlagEmbedding import FlagModel
    except ImportError as exc:
        raise RuntimeError("请先安装 FlagEmbedding==1.2.11") from exc
    except Exception as exc:
        raise RuntimeError(
            "导入 FlagEmbedding 失败，常见原因是安装了存在 typing.Optional 导入缺失问题的版本。"
            "建议升级/降级 FlagEmbedding，或继续使用当前文件内的兼容补丁。"
        ) from exc

    model_name = os.environ.get("EMBEDDING_MODEL", "/root/models/BAAI/bge-m3")
    device = os.environ.get("EMBEDDING_DEVICE", "cuda:0")
    use_fp16 = "cuda" in device

    init_kwargs: dict[str, Any] = {"use_fp16": use_fp16}

    logger.info("加载 embedding 模型: %s device=%s fp16=%s", model_name, device, use_fp16)
    _model = FlagModel(model_name, **init_kwargs)
    logger.info("embedding 模型加载完成")
    return _model


@app.get("/health")
def health():
    return {"status": "ok", "model": os.environ.get("EMBEDDING_MODEL", "/root/models/BAAI/bge-m3")}


@app.post("/v1/embeddings", dependencies=[Depends(_check_auth)])
def embeddings(body: EmbeddingRequest):
    model = _get_model()
    inputs = body.input if isinstance(body.input, list) else [body.input]
    vectors = model.encode(inputs, batch_size=min(len(inputs), 32))
    data = [
        {"object": "embedding", "index": i, "embedding": vec.tolist()}
        for i, vec in enumerate(vectors)
    ]
    prompt_tokens = sum(max(1, len(text) // 4) for text in inputs)
    return {
        "object": "list",
        "data": data,
        "model": body.model,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "total_tokens": prompt_tokens,
        },
    }
