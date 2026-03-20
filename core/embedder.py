"""
core/embedder.py — 向量化 + Milvus 落库。

.env 配置：
    EMBEDDING_MODEL=BAAI/bge-m3   # HuggingFace 名称或本地路径
    EMBEDDING_DEVICE=cpu           # cpu / cuda / cuda:1
    EMBEDDING_BATCH_SIZE=32
    MILVUS_HOST=localhost
    MILVUS_PORT=19530
    MILVUS_COLLECTION=rag_docs

Milvus collection schema：
    pk          INT64    主键（auto_id）
    chunk_id    VARCHAR  chunk 内容 SHA256（去重用）
    file_id     VARCHAR  文件 SHA256（按文件过滤/删除用）
    chunk_index INT64    块序号
    source      VARCHAR  原始文件路径
    label       VARCHAR  OCR 块类型（ocr/text/table 等）
    content     VARCHAR  chunk 原文
    vector      FLOAT_VECTOR(1024)

使用示例：
    vectors = Embedder.embed(chunks)
    Embedder.store(chunks, vectors, file_id, hashes)
    results = Embedder.search("查询文本", top_k=5)
    query_vec = Embedder.embed_query("查询文本")
"""

from __future__ import annotations

import logging
from typing import Any

from config.settings import settings
from models.document import Document

logger = logging.getLogger(__name__)

_DIM = 1024  # BGE-M3 固定维度

# ---------------------------------------------------------------------------
# BGE-M3 模型单例
# ---------------------------------------------------------------------------

_model: Any | None = None


def _get_model():
    global _model
    if _model is None:
        try:
            from FlagEmbedding import FlagModel
        except ImportError as exc:
            raise ImportError(
                "FlagEmbedding 未安装: pip install FlagEmbedding==1.2.11"
            ) from exc

        device   = settings.embedding_device
        use_fp16 = "cuda" in device

        logger.info("[Embedder] 加载: %s  device=%s  fp16=%s",
                    settings.embedding_model, device, use_fp16)

        init_kwargs: dict = {"use_fp16": use_fp16}
        if device != "cpu":
            init_kwargs["device"] = device

        _model = FlagModel(settings.embedding_model, **init_kwargs)
        logger.info("[Embedder] 加载完成")
    return _model


def reset_model() -> None:
    global _model
    _model = None


# ---------------------------------------------------------------------------
# Milvus collection 单例
# ---------------------------------------------------------------------------

_collection: Any | None = None


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
        raise ImportError("pymilvus 未安装: pip install pymilvus") from exc

    connections.connect(host=settings.milvus_host, port=settings.milvus_port)
    logger.info("[Milvus] 连接: %s:%s", settings.milvus_host, settings.milvus_port)

    col_name = settings.milvus_collection

    if not utility.has_collection(col_name):
        schema = CollectionSchema(
            fields=[
                FieldSchema("pk",          DataType.INT64,        is_primary=True, auto_id=True),
                FieldSchema("chunk_id",    DataType.VARCHAR,      max_length=64),
                FieldSchema("file_id",     DataType.VARCHAR,      max_length=64),
                FieldSchema("chunk_index", DataType.INT64),
                FieldSchema("source",      DataType.VARCHAR,      max_length=512),
                FieldSchema("label",       DataType.VARCHAR,      max_length=64),
                FieldSchema("content",     DataType.VARCHAR,      max_length=65535),
                FieldSchema("vector",      DataType.FLOAT_VECTOR, dim=_DIM),
            ],
            description="RAG document chunks",
            enable_dynamic_field=False,
        )
        _collection = Collection(name=col_name, schema=schema)
        _collection.create_index(
            field_name="vector",
            index_params={
                "metric_type": "COSINE",
                "index_type":  "HNSW",
                "params":      {"M": 16, "efConstruction": 200},
            },
        )
        logger.info("[Milvus] 集合已创建: %s", col_name)
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
        """批量向量化，返回向量列表（顺序与输入一致）。"""
        if not chunks:
            return []

        bs    = batch_size or settings.embedding_batch_size
        texts = [c.page_content for c in chunks]

        logger.info("[Embedder] 向量化 %d 个 chunk，batch_size=%d", len(texts), bs)
        vectors = _get_model().encode(texts, batch_size=bs)
        result  = vectors.tolist()
        logger.info("[Embedder] 完成，维度=%d", len(result[0]) if result else 0)
        logger.debug("[Embedder] 前3向量前5维: %s",
                     [[round(v, 4) for v in vec[:5]] for vec in result[:3]])
        return result

    @classmethod
    def embed_query(cls, query: str) -> list[float]:
        """对单条查询文本向量化（搜索时使用）。"""
        logger.debug("[Embedder] 向量化查询: %r", query[:60])
        vectors = _get_model().encode_queries([query])
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
        将 chunk + 向量写入 Milvus，返回实际写入条数。

        Parameters
        ----------
        chunks       : 经去重过滤后的新 chunk
        vectors      : Embedder.embed() 输出，顺序与 chunks 一致
        file_id      : 文件 SHA256（Dedup.register_file 返回值）
        chunk_hashes : chunk 内容 SHA256（Dedup.filter_new_chunks 返回值）
        """
        if not chunks:
            logger.info("[Milvus] 无新 chunk，跳过写入")
            return 0

        assert len(chunks) == len(vectors) == len(chunk_hashes), \
            "chunks/vectors/hashes 长度不一致"

        MAX_CONTENT = 60000
        rows = [
            {
                "chunk_id":    h,
                "file_id":     file_id,
                "chunk_index": chunk.metadata.get("chunk_index", i),
                "source":      chunk.metadata.get("source", "")[:512],
                "label":       chunk.metadata.get("label", "unknown")[:64],
                "content":     chunk.page_content[:MAX_CONTENT],
                "vector":      vec,
            }
            for i, (chunk, vec, h) in enumerate(zip(chunks, vectors, chunk_hashes))
        ]

        col = _get_collection()
        col.insert(rows)
        col.flush()
        logger.info("[Milvus] 写入 %d 条", len(rows))
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
        向量相似搜索。

        Parameters
        ----------
        query           : 查询文本
        top_k           : 返回条数
        score_threshold : 最低相似度（COSINE 0~1）
        filter_expr     : Milvus 过滤表达式，如 'file_id == "abc"'

        Returns
        -------
        list[dict]，每条含 id/file_id/source/label/content/score
        """
        query_vec = cls.embed_query(query)
        col = _get_collection()

        results = col.search(
            data=[query_vec],
            anns_field="vector",
            param={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=top_k,
            expr=filter_expr,
            output_fields=["chunk_id", "file_id", "source", "label", "content"],
        )

        hits = []
        for hit in results[0]:
            if hit.score < score_threshold:
                continue
            hits.append({
                "id":      hit.entity.get("chunk_id"),
                "file_id": hit.entity.get("file_id"),
                "source":  hit.entity.get("source"),
                "label":   hit.entity.get("label"),
                "content": hit.entity.get("content"),
                "score":   round(hit.score, 4),
            })

        logger.info("[Milvus] 搜索 %r → %d 条结果", query[:30], len(hits))
        logger.debug("[Milvus] 搜索结果: %s", hits)
        return hits

    # ------------------------------------------------------------------
    #  管理
    # ------------------------------------------------------------------

    @classmethod
    def delete_by_file(cls, file_id: str) -> None:
        """删除某文件的全部向量。"""
        col = _get_collection()
        col.delete(expr=f'file_id == "{file_id}"')
        col.flush()
        logger.info("[Milvus] 删除文件向量: file_id=%s", file_id[:12])

    @classmethod
    def count(cls) -> int:
        """返回集合中的向量总数。"""
        return _get_collection().num_entities
        