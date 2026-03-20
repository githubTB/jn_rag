"""
core/embedder.py — 向量化 + Milvus 落库。

本地加载 BGE-M3，支持 CPU / GPU 自动切换。
Milvus collection 首次调用时自动建立。

settings 相关配置（.env）：
    EMBEDDING_MODEL=BAAI/bge-m3        # HuggingFace 名称 或 本地绝对路径
    EMBEDDING_DEVICE=cpu               # cpu / cuda / cuda:1
    EMBEDDING_BATCH_SIZE=32
    MILVUS_HOST=localhost
    MILVUS_PORT=19530
    MILVUS_COLLECTION=rag_docs

使用示例
--------
    from core.embedder import Embedder

    # 向量化
    vectors = Embedder.embed(chunks)           # list[list[float]]
    query_vec = Embedder.embed_query("查询文本") # list[float]

    # 落库（向量 + chunk 一一对应）
    Embedder.store(chunks, vectors, file_id, hashes)

    # 搜索
    results = Embedder.search("查询文本", top_k=5)
"""

from __future__ import annotations

import logging
from typing import Any

from config.settings import settings
from models.document import Document

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 模型单例
# ---------------------------------------------------------------------------

_model: Any | None = None


def _get_model():
    global _model
    if _model is None:
        try:
            from FlagEmbedding import FlagModel
        except ImportError as exc:
            raise ImportError(
                "FlagEmbedding 未安装: pip install FlagEmbedding"
            ) from exc

        device = settings.embedding_device  # "cpu" / "cuda" / "cuda:1"
        use_fp16 = "cuda" in device

        logger.info(
            "[Embedder] 加载模型: %s  device=%s  fp16=%s",
            settings.embedding_model, device, use_fp16,
        )
        _model = FlagModel(
            settings.embedding_model,
            use_fp16=use_fp16,
            devices=[device] if device != "cpu" else None,
        )
        logger.info("[Embedder] 模型加载完成")
    return _model


def reset_model() -> None:
    global _model
    _model = None


# ---------------------------------------------------------------------------
# Milvus 集合单例
# ---------------------------------------------------------------------------

_collection: Any | None = None

# Collection schema 字段说明：
#   id          : chunk SHA256（主键，去重用）
#   file_id     : 所属文件 SHA256（按文件过滤/删除用）
#   chunk_index : 块序号
#   source      : 原始文件路径
#   label       : OCR 块类型（ocr / text / table 等）
#   content     : chunk 原文（搜索结果展示用）
#   vector      : BGE-M3 向量（1024 维）

_DIM = 1024  # BGE-M3 固定维度


def _get_collection():
    global _collection
    if _collection is not None:
        return _collection

    try:
        from pymilvus import (
            connections, Collection, CollectionSchema,
            FieldSchema, DataType, utility,
        )
    except ImportError as exc:
        raise ImportError(
            "pymilvus 未安装: pip install pymilvus"
        ) from exc

    # 连接
    connections.connect(
        host=settings.milvus_host,
        port=settings.milvus_port,
    )
    logger.info(
        "[Milvus] 连接: %s:%s", settings.milvus_host, settings.milvus_port
    )

    col_name = settings.milvus_collection

    # 集合不存在则创建
    if not utility.has_collection(col_name):
        schema = CollectionSchema(
            fields=[
                FieldSchema("id",          DataType.VARCHAR,       max_length=64,   is_primary=True),
                FieldSchema("file_id",     DataType.VARCHAR,       max_length=64),
                FieldSchema("chunk_index", DataType.INT64),
                FieldSchema("source",      DataType.VARCHAR,       max_length=512),
                FieldSchema("label",       DataType.VARCHAR,       max_length=64),
                FieldSchema("content",     DataType.VARCHAR,       max_length=65535),
                FieldSchema("vector",      DataType.FLOAT_VECTOR,  dim=_DIM),
            ],
            description="RAG document chunks",
            enable_dynamic_field=False,
        )
        _collection = Collection(name=col_name, schema=schema)
        logger.info("[Milvus] 创建集合: %s", col_name)

        # 创建向量索引（HNSW，召回率和速度均衡）
        _collection.create_index(
            field_name="vector",
            index_params={
                "metric_type": "COSINE",
                "index_type":  "HNSW",
                "params":      {"M": 16, "efConstruction": 200},
            },
        )
        logger.info("[Milvus] 向量索引创建完成")
    else:
        _collection = Collection(name=col_name)
        logger.debug("[Milvus] 使用已有集合: %s", col_name)

    _collection.load()
    return _collection


# ---------------------------------------------------------------------------
# 工具类
# ---------------------------------------------------------------------------


class Embedder:
    """
    向量化 + Milvus 落库工具类，所有方法均为类方法，无需实例化。

    典型链路：
        vectors = Embedder.embed(new_chunks)
        Embedder.store(new_chunks, vectors, file_id, new_hashes)
    """

    # ------------------------------------------------------------------
    #  向量化
    # ------------------------------------------------------------------

    @classmethod
    def embed(
        cls,
        chunks: list[Document],
        batch_size: int | None = None,
    ) -> list[list[float]]:
        """
        批量向量化 chunk 列表，返回向量列表（顺序与输入一致）。
        """
        if not chunks:
            return []

        bs = batch_size or settings.embedding_batch_size
        texts = [c.page_content for c in chunks]

        logger.info("[Embedder] 向量化 %d 个 chunk，batch_size=%d", len(texts), bs)

        model = _get_model()
        vectors = model.encode(texts, batch_size=bs)
        result = vectors.tolist()

        logger.info(
            "[Embedder] 向量化完成，维度=%d", len(result[0]) if result else 0
        )
        logger.debug(
            "[Embedder] 前3个向量前5维: %s",
            [[round(v, 4) for v in vec[:5]] for vec in result[:3]],
        )
        return result

    @classmethod
    def embed_query(cls, query: str) -> list[float]:
        """对单条查询文本向量化（搜索时使用）。"""
        logger.debug("[Embedder] 向量化查询: %r", query[:60])
        model = _get_model()
        vectors = model.encode_queries([query])
        return vectors[0].tolist()

    @classmethod
    def dim(cls) -> int:
        return _DIM

    # ------------------------------------------------------------------
    #  Milvus 落库
    # ------------------------------------------------------------------

    @classmethod
    def store(
        cls,
        chunks: list[Document],
        vectors: list[list[float]],
        file_id: str,
        chunk_hashes: list[str],
    ) -> int:
        """
        将 chunk + 向量写入 Milvus。

        Parameters
        ----------
        chunks       : DocChunker.chunk() 输出（经去重过滤后的新 chunk）
        vectors      : Embedder.embed() 输出，顺序与 chunks 一致
        file_id      : 文件 SHA256（来自 Dedup.register_file）
        chunk_hashes : chunk 内容 SHA256 列表（来自 Dedup.filter_new_chunks）

        Returns
        -------
        int : 实际写入条数
        """
        if not chunks:
            logger.info("[Milvus] 无新 chunk，跳过写入")
            return 0

        assert len(chunks) == len(vectors) == len(chunk_hashes), \
            "chunks / vectors / hashes 长度不一致"

        col = _get_collection()

        # 截断 content 防止超过 VARCHAR(65535)
        MAX_CONTENT = 60000

        rows = [
            {
                "id":          h,
                "file_id":     file_id,
                "chunk_index": chunk.metadata.get("chunk_index", i),
                "source":      chunk.metadata.get("source", "")[:512],
                "label":       chunk.metadata.get("label", "unknown")[:64],
                "content":     chunk.page_content[:MAX_CONTENT],
                "vector":      vec,
            }
            for i, (chunk, vec, h) in enumerate(zip(chunks, vectors, chunk_hashes))
        ]

        col.insert(rows)
        col.flush()

        logger.info("[Milvus] 写入 %d 条 chunk", len(rows))
        return len(rows)

    # ------------------------------------------------------------------
    #  向量搜索
    # ------------------------------------------------------------------

    @classmethod
    def search(
        cls,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.3,
        filter_expr: str | None = None,
    ) -> list[dict]:
        """
        向量相似搜索，返回最相关的 chunk 列表。

        Parameters
        ----------
        query          : 用户查询文本
        top_k          : 返回条数
        score_threshold: 最低相似度（COSINE，0~1），低于此值过滤掉
        filter_expr    : Milvus 过滤表达式，例如 'file_id == "abc123"'

        Returns
        -------
        list[dict]，每条含 id / source / label / content / score
        """
        query_vec = cls.embed_query(query)
        col = _get_collection()

        search_params = {"metric_type": "COSINE", "params": {"ef": 64}}
        results = col.search(
            data=[query_vec],
            anns_field="vector",
            param=search_params,
            limit=top_k,
            expr=filter_expr,
            output_fields=["id", "file_id", "source", "label", "content"],
        )

        hits = []
        for hit in results[0]:
            score = hit.score
            if score < score_threshold:
                continue
            hits.append({
                "id":      hit.id,
                "file_id": hit.entity.get("file_id"),
                "source":  hit.entity.get("source"),
                "label":   hit.entity.get("label"),
                "content": hit.entity.get("content"),
                "score":   round(score, 4),
            })

        logger.info(
            "[Milvus] 搜索 %r → %d 条结果（top_k=%d）", query[:30], len(hits), top_k
        )
        logger.debug("[Milvus] 搜索结果: %s", hits)
        return hits

    # ------------------------------------------------------------------
    #  管理
    # ------------------------------------------------------------------

    @classmethod
    def delete_by_file(cls, file_id: str) -> None:
        """删除某个文件的所有 chunk（重新处理时使用）。"""
        col = _get_collection()
        col.delete(expr=f'file_id == "{file_id}"')
        col.flush()
        logger.info("[Milvus] 删除文件 chunk: file_id=%s", file_id[:12])

    @classmethod
    def count(cls) -> int:
        """返回集合中的 chunk 总数。"""
        col = _get_collection()
        return col.num_entities
        