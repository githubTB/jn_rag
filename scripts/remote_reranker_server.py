"""
scripts/remote_reranker_server.py - 远程 Reranker 服务。

提供 OpenAI 兼容风格的重排接口，供本项目的 remote Reranker 调用。

启动示例：
    RERANKER_MODEL=BAAI/bge-reranker-v2-m3 \
    RERANKER_DEVICE=cuda:0 \
    RERANKER_API_KEY=EMPTY \
    uvicorn scripts.remote_reranker_server:app --host 0.0.0.0 --port 8002
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

app = FastAPI(title="Remote Reranker", version="0.1.0")

_model: Any | None = None


class RerankRequest(BaseModel):
    model: str
    query: str
    documents: list[str]
    top_n: int | None = None


def _check_auth(authorization: str | None = Header(default=None)) -> None:
    expected = os.environ.get("RERANKER_API_KEY", "EMPTY")
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
        from FlagEmbedding import FlagReranker
    except ImportError as exc:
        raise RuntimeError("请先安装 FlagEmbedding==1.2.11") from exc
    except Exception as exc:
        raise RuntimeError(
            "导入 FlagEmbedding 失败，常见原因是安装了存在 typing.Optional 导入缺失问题的版本。"
            "建议升级/降级 FlagEmbedding，或继续使用当前文件内的兼容补丁。"
        ) from exc

    model_name = os.environ.get("RERANKER_MODEL", "/root/models/BAAI/bge-reranker-v2-m3")
    device = os.environ.get("RERANKER_DEVICE", "cuda:0")
    use_fp16 = "cuda" in device

    init_kwargs: dict[str, Any] = {"use_fp16": use_fp16}

    logger.info("加载 reranker 模型: %s device=%s fp16=%s", model_name, device, use_fp16)
    _model = FlagReranker(model_name, **init_kwargs)
    logger.info("reranker 模型加载完成")
    return _model


@app.get("/health")
def health():
    return {"status": "ok", "model": os.environ.get("RERANKER_MODEL", "/root/models/BAAI/bge-reranker-v2-m3")}


@app.post("/v1/rerank", dependencies=[Depends(_check_auth)])
def rerank(body: RerankRequest):
    model = _get_model()

    if not body.documents:
        return {"object": "list", "data": [], "model": body.model}

    pairs = [[body.query, doc] for doc in body.documents]
    scores = model.compute_score(pairs)
    if not isinstance(scores, list):
        scores = [scores]

    results = [
        {
            "object": "rerank",
            "index": idx,
            "document": {"text": doc},
            "relevance_score": float(score),
        }
        for idx, (doc, score) in enumerate(zip(body.documents, scores))
    ]
    results.sort(key=lambda item: item["relevance_score"], reverse=True)

    if body.top_n is not None:
        results = results[: body.top_n]

    return {
        "object": "list",
        "data": results,
        "model": body.model,
        "usage": {
            "total_tokens": max(1, len(body.query) // 4)
            + sum(max(1, len(doc) // 4) for doc in body.documents),
        },
    }
