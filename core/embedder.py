"""
core/embedder.py — 向量化模块（BGE-M3）。

设计原则
--------
与现有代码风格保持一致：
  - ExtractProcessor.extract()   → 类方法静态调用
  - DocChunker.chunk()           → 类方法静态调用
  - Embedder.embed()             → 类方法静态调用  ← 本模块
  - DedupStore()                 → 实例调用（有状态，需要数据库连接）

模型单例：BGE-M3 权重约 2.2GB，进程内只加载一次，
通过模块级 _model 缓存，线程安全（只写一次）。

完整链路示例
-----------
    from extract_processor import ExtractProcessor
    from extractor.doc_chunker import DocChunker
    from core.dedup import DedupStore
    from core.embedder import Embedder

    # 1. 解析
    docs = ExtractProcessor.extract("invoice.jpg")

    # 2. 切片
    chunks = DocChunker.chunk(docs)

    # 3. 去重
    store = DedupStore()
    file_hash = DedupStore.hash_file("invoice.jpg")
    new_chunks, new_hashes = store.filter_new_chunks(chunks, file_hash)

    # 4. 向量化
    vectors = Embedder.embed(new_chunks)

    # 5. vectors 和 new_chunks 一一对应，直接送 Milvus
"""

from __future__ import annotations

import logging
from typing import Any

from config.settings import settings
from models.document import Document

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 模型单例（进程内只加载一次）
# ---------------------------------------------------------------------------

_model: Any | None = None


def _get_model():
    """返回全局 FlagModel 实例，首次调用时初始化。"""
    global _model
    if _model is None:
        try:
            from FlagEmbedding import FlagModel
        except ImportError as exc:
            raise ImportError(
                "FlagEmbedding 未安装，请执行: pip install FlagEmbedding"
            ) from exc

        logger.info(
            "[Embedder] 加载模型: %s  device: %s",
            settings.embedding_model,
            settings.embedding_device,
        )
        _model = FlagModel(
            settings.embedding_model,
            use_fp16=settings.embedding_device != "cpu",  # CPU 不支持 fp16
            devices=settings.embedding_device if settings.embedding_device != "cpu" else None,
        )
        logger.info("[Embedder] 模型加载完成")
    return _model


def reset_model() -> None:
    """强制下次调用时重新加载模型（测试用）。"""
    global _model
    _model = None


# ---------------------------------------------------------------------------
# 主类
# ---------------------------------------------------------------------------


class Embedder:
    """
    BGE-M3 向量化工具类。

    所有方法均为类方法，无需实例化，与 ExtractProcessor / DocChunker 风格一致。
    """

    @classmethod
    def embed(
        cls,
        chunks: list[Document],
        batch_size: int | None = None,
    ) -> list[list[float]]:
        """
        对 chunk 列表批量向量化，返回向量列表。

        返回顺序与输入 chunks 一一对应，可直接与 chunk 列表 zip 使用。

        Parameters
        ----------
        chunks : list[Document]
            DocChunker.chunk() 的输出，或经 DedupStore.filter_new_chunks() 过滤后的列表。
        batch_size : int | None
            批次大小，None 时使用 settings.embedding_batch_size（默认 32）。

        Returns
        -------
        list[list[float]]
            每个元素是一个 1024 维的 float 向量（BGE-M3 默认维度）。
        """
        if not chunks:
            return []

        bs = batch_size or settings.embedding_batch_size
        texts = [c.page_content for c in chunks]

        logger.info(
            "[Embedder] 开始向量化: %d 个 chunk，batch_size=%d",
            len(texts), bs,
        )

        model = _get_model()
        vectors = model.encode(texts, batch_size=bs)

        # FlagModel.encode 返回 numpy array，转成 list[list[float]] 方便序列化
        result = vectors.tolist()

        logger.info("[Embedder] 向量化完成，维度: %d", len(result[0]) if result else 0)
        return result

    @classmethod
    def embed_query(cls, query: str) -> list[float]:
        """
        对单条查询文本向量化（搜索时使用）。

        与 embed() 分开是因为 BGE-M3 对 query 和 passage 的
        encode 策略略有不同（query 不加 instruction）。
        """
        logger.debug("[Embedder] 向量化查询: %r", query[:50])
        model = _get_model()
        vector = model.encode_queries([query])
        return vector[0].tolist()

    @classmethod
    def embed_texts(cls, texts: list[str], batch_size: int | None = None) -> list[list[float]]:
        """
        直接对文本列表向量化（不需要 Document 对象时使用）。
        """
        if not texts:
            return []
        fake_docs = [Document(page_content=t, metadata={}) for t in texts]
        return cls.embed(fake_docs, batch_size=batch_size)

    @classmethod
    def dim(cls) -> int:
        """返回向量维度（BGE-M3 为 1024）。"""
        return 1024
