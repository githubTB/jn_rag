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
import os
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 模型单例
# ---------------------------------------------------------------------------

_reranker: Any | None = None


def _get_reranker():
    global _reranker
    if _reranker is not None:
        return _reranker

    model_name = os.environ.get("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
    device     = os.environ.get("RERANKER_DEVICE", "cpu")
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
    return os.environ.get("RERANKER_ENABLED", "true").lower() not in ("false", "0", "no")


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
            reranker = _get_reranker()
        except Exception as exc:
            logger.warning("[Reranker] 加载失败，降级返回向量检索结果: %s", exc)
            return hits[:top_n]

        # 构造 (query, passage) 对
        pairs = [[query, h.get("content", "")] for h in hits]

        try:
            scores = reranker.compute_score(pairs, normalize=True)
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
            _get_reranker()
            return True
        except Exception:
            return False