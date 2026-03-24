"""
core/embedder.py — 向量化 + Milvus 落库。

Milvus collection schema：
    pk          INT64    主键（auto_id）
    chunk_id    VARCHAR  chunk 内容 SHA256（去重用）
    file_id     VARCHAR  文件 SHA256
    company_id  VARCHAR  企业 ID
    doc_type    VARCHAR  文件类型
    chunk_index INT64    块序号
    source      VARCHAR  原始文件路径
    label       VARCHAR  块类型（ocr/text/table 等）
    content     VARCHAR  纯文本内容（用于向量检索）
    raw_content VARCHAR  原始内容（保留格式，用于报告生成）
    vector      FLOAT_VECTOR(1024)
"""

from __future__ import annotations

import logging
from typing import Any

from config.settings import settings
from models.document import Document

logger = logging.getLogger(__name__)

_DIM = 1024

_model: Any | None = None


def _get_model():
    if settings.embedding_provider.lower() == "remote":
        return None

    global _model
    if _model is None:
        try:
            from FlagEmbedding import FlagModel
        except ImportError as exc:
            raise ImportError("FlagEmbedding 未安装: pip install FlagEmbedding==1.2.11") from exc

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
                FieldSchema("company_id",  DataType.VARCHAR,      max_length=64),
                FieldSchema("doc_type",    DataType.VARCHAR,      max_length=32),
                FieldSchema("chunk_index", DataType.INT64),
                FieldSchema("source",      DataType.VARCHAR,      max_length=512),
                FieldSchema("label",       DataType.VARCHAR,      max_length=64),
                FieldSchema("content",     DataType.VARCHAR,      max_length=65535),
                FieldSchema("raw_content", DataType.VARCHAR,      max_length=65535),
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


def _html_to_plain(html: str) -> str:
    """HTML 表格转纯文本，用于向量检索。"""
    if "<table" not in html:
        return html
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        rows = []
        for tr in soup.find_all("tr"):
            cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
            if any(cells):
                rows.append(" | ".join(cells))
        return "\n".join(rows)
    except Exception:
        import re
        return re.sub(r"<[^>]+>", " ", html).strip()


class Embedder:

    @classmethod
    def embed(
        cls,
        chunks: list[Document],
        batch_size: int | None = None,
    ) -> list[list[float]]:
        if not chunks:
            return []
        bs    = batch_size or settings.embedding_batch_size
        # 向量化用纯文本内容（已去掉HTML标签）
        texts = [_html_to_plain(c.page_content) for c in chunks]
        logger.info("[Embedder] 向量化 %d 个 chunk，batch_size=%d", len(texts), bs)
        if settings.embedding_provider.lower() == "remote":
            result = cls._embed_remote(texts)
        else:
            vectors = _get_model().encode(texts, batch_size=bs)
            result  = vectors.tolist()
        logger.info("[Embedder] 完成，维度=%d", len(result[0]) if result else 0)
        return result

    @classmethod
    def embed_query(cls, query: str) -> list[float]:
        if settings.embedding_provider.lower() == "remote":
            return cls._embed_remote([query])[0]
        vectors = _get_model().encode_queries([query])
        return vectors[0].tolist()

    @classmethod
    def dim(cls) -> int:
        return _DIM

    @classmethod
    def store(
        cls,
        chunks: list[Document],
        vectors: list[list[float]],
        file_id: str,
        chunk_hashes: list[str],
        company_id: str | None = None,
        doc_type: str = "unknown",
    ) -> int:
        if not chunks:
            logger.info("[Milvus] 无新 chunk，跳过写入")
            return 0

        assert len(chunks) == len(vectors) == len(chunk_hashes), \
            "chunks/vectors/hashes 长度不一致"

        MAX_CONTENT = 60000
        rows = []
        for i, (chunk, vec, h) in enumerate(zip(chunks, vectors, chunk_hashes)):
            raw = chunk.page_content
            # content 存纯文本（用于检索），raw_content 存原始内容（用于报告生成）
            plain = _html_to_plain(raw)
            rows.append({
                "chunk_id":    h,
                "file_id":     file_id,
                "company_id":  company_id or "",
                "doc_type":    doc_type[:32],
                "chunk_index": chunk.metadata.get("chunk_index", i),
                "source":      chunk.metadata.get("source", "")[:512],
                "label":       chunk.metadata.get("label", "unknown")[:64],
                "content":     plain[:MAX_CONTENT],
                "raw_content": raw[:MAX_CONTENT],
                "vector":      vec,
            })

        col = _get_collection()
        col.insert(rows)
        col.flush()
        logger.info("[Milvus] 写入 %d 条", len(rows))
        return len(rows)

    @classmethod
    def search(
        cls,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.3,
        filter_expr: str | None = None,
    ) -> list[dict]:
        query_vec = cls.embed_query(query)
        col = _get_collection()

        results = col.search(
            data=[query_vec],
            anns_field="vector",
            param={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=top_k,
            expr=filter_expr,
            output_fields=[
                "chunk_id", "file_id", "company_id", "doc_type",
                "source", "label", "content", "raw_content",
            ],
        )

        hits = []
        for hit in results[0]:
            if hit.score < score_threshold:
                continue
            hits.append({
                "id":          hit.entity.get("chunk_id"),
                "file_id":     hit.entity.get("file_id"),
                "company_id":  hit.entity.get("company_id"),
                "doc_type":    hit.entity.get("doc_type"),
                "source":      hit.entity.get("source"),
                "label":       hit.entity.get("label"),
                "content":     hit.entity.get("content"),
                "raw_content": hit.entity.get("raw_content"),
                "score":       round(hit.score, 4),
            })

        logger.info("[Milvus] 搜索 %r → %d 条结果", query[:30], len(hits))
        return hits

    @classmethod
    def delete_by_file(cls, file_id: str) -> None:
        col = _get_collection()
        col.delete(expr=f'file_id == "{file_id}"')
        col.flush()
        logger.info("[Milvus] 删除文件向量: file_id=%s", file_id[:12])

    @classmethod
    def count(cls) -> int:
        return _get_collection().num_entities

    @classmethod
    def _embed_remote(cls, texts: list[str]) -> list[list[float]]:
        if not settings.embedding_api_base:
            raise RuntimeError("EMBEDDING_PROVIDER=remote 时必须配置 EMBEDDING_API_BASE")

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("远程 embedding 需要安装 openai: pip install openai") from exc

        client = OpenAI(
            base_url=settings.embedding_api_base,
            api_key=settings.embedding_api_key or "EMPTY",
        )

        response = client.embeddings.create(
            model=settings.embedding_model,
            input=texts,
        )
        return [item.embedding for item in response.data]
