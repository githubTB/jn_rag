"""
core/reranker.py — 检索结果重排序（Cross-encoder Reranker）。

使用 BAAI/bge-reranker-v2-m3，跑在 CPU，不占显存。
对向量检索的粗筛结果做精排，显著提升 RAG 召回质量。

.env 配置：
    RERANKER_MODEL=BAAI/bge-reranker-v2-m3   # 本地路径或 HuggingFace 名称
    RERANKER_DEVICE=cpu                        # cpu / cuda
    RERANKER_ENABLED=true                      # false 可临时关闭

使用示例：
    from core.reranker import Reranker

    hits = Embedder.search(query, top_k=20)
    hits = Reranker.rerank(query, hits, top_n=5)
"""

from __future__ import annotations

import logging
import urllib.request
import json
from typing import Any

from config.settings import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 模型单例
# ---------------------------------------------------------------------------

_reranker: Any | None = None


def _get_reranker():
    if settings.reranker_provider.lower() == "remote":
        return None

    global _reranker
    if _reranker is not None:
        return _reranker

    model_name = settings.reranker_model
    device     = settings.reranker_device
    use_fp16   = "cuda" in device

    try:
        from FlagEmbedding import FlagReranker
    except ImportError as exc:
        raise ImportError(
            "FlagEmbedding 未安装: pip install FlagEmbedding==1.2.11"
        ) from exc

    logger.info("[Reranker] 加载: %s  device=%s  fp16=%s", model_name, device, use_fp16)
    _reranker = FlagReranker(model_name, use_fp16=use_fp16)
    logger.info("[Reranker] 加载完成")
    return _reranker


def reset_reranker() -> None:
    """强制下次调用时重新加载（切换模型或测试用）。"""
    global _reranker
    _reranker = None
    logger.info("[Reranker] 已重置")


def _is_enabled() -> bool:
    return str(settings.reranker_enabled).lower() not in ("false", "0", "no")


# ---------------------------------------------------------------------------
# 工具类
# ---------------------------------------------------------------------------

class Reranker:
    """
    检索结果重排序工具类，所有方法均为类方法，无需实例化。

    典型链路：
        hits = Embedder.search(query, top_k=20, score_threshold=0.2)
        hits = Reranker.rerank(query, hits, top_n=5)
        # hits 已按 rerank_score 降序排列，只保留前 top_n 条
    """

    @classmethod
    def rerank(
        cls,
        query: str,
        hits: list[dict],
        top_n: int = 5,
    ) -> list[dict]:
        """
        对向量检索结果做 Cross-encoder 精排。

        Parameters
        ----------
        query  : 原始查询文本
        hits   : Embedder.search() 返回的结果列表
        top_n  : 精排后保留的条数

        Returns
        -------
        按 rerank_score 降序排列的 top_n 条结果。
        每条结果新增 rerank_score 字段（0~1，越高越相关）。
        Reranker 未启用时原样返回前 top_n 条（保留原向量分数）。
        """
        if not hits:
            return hits

        # Reranker 未启用时降级：直接截断
        if not _is_enabled():
            logger.debug("[Reranker] 已禁用，跳过精排")
            return hits[:top_n]

        try:
            scores = cls._compute_scores(query, hits)
        except Exception as exc:
            logger.warning("[Reranker] 打分失败，降级返回向量检索结果: %s", exc)
            return hits[:top_n]

        # 写回分数
        if isinstance(scores, (int, float)):
            scores = [scores]
        for hit, score in zip(hits, scores):
            hit["rerank_score"] = round(float(score), 4)

        # 按 rerank_score 降序排列，取 top_n
        reranked = sorted(hits, key=lambda x: x.get("rerank_score", 0), reverse=True)
        result   = reranked[:top_n]

        logger.info(
            "[Reranker] 精排完成: %d → %d  top1_score=%.4f",
            len(hits), len(result),
            result[0]["rerank_score"] if result else 0,
        )
        logger.debug(
            "[Reranker] 精排结果: %s",
            [(r.get("source","")[-30:], r.get("rerank_score")) for r in result],
        )
        return result

    @classmethod
    def is_available(cls) -> bool:
        """检查 Reranker 是否可用（模型可加载且已启用）。"""
        if not _is_enabled():
            return False
        try:
            if settings.reranker_provider.lower() == "remote":
                return bool(settings.reranker_api_base)
            _get_reranker()
            return True
        except Exception:
            return False

    @classmethod
    def _compute_scores(cls, query: str, hits: list[dict]) -> list[float]:
        if settings.reranker_provider.lower() == "remote":
            return cls._compute_scores_remote(query, hits)

        try:
            reranker = _get_reranker()
        except Exception as exc:
            raise RuntimeError(f"加载本地 Reranker 失败: {exc}") from exc

        pairs = [[query, h.get("content", "")] for h in hits]
        scores = reranker.compute_score(pairs, normalize=True)
        if isinstance(scores, (int, float)):
            return [float(scores)]
        return [float(s) for s in scores]

    @classmethod
    def _compute_scores_remote(cls, query: str, hits: list[dict]) -> list[float]:
        if not settings.reranker_api_base:
            raise RuntimeError("RERANKER_PROVIDER=remote 时必须配置 RERANKER_API_BASE")

        url = settings.reranker_api_base.rstrip("/") + "/rerank"
        payload = {
            "model": settings.reranker_model,
            "query": query,
            "documents": [h.get("content", "") for h in hits],
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                **({"Authorization": f"Bearer {settings.reranker_api_key}"} if settings.reranker_api_key else {}),
            },
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        if isinstance(data, dict) and "results" in data:
            return [float(item.get("relevance_score", item.get("score", 0))) for item in data["results"]]
        if isinstance(data, dict) and "data" in data:
            return [float(item.get("relevance_score", item.get("score", 0))) for item in data["data"]]
        raise RuntimeError(f"远程 rerank 返回格式不支持: {data}")
