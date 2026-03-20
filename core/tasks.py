"""
core/tasks.py — 完整入库流水线。

串联：文件去重 → 解析 → 切片 → Chunk 去重 → 向量化 → 落库 → 标记完成

设计原则
--------
- 每步失败都有明确日志和状态标记（Dedup.mark_failed）
- 文件级去重在解析前，Chunk 级去重在向量化前
- 幂等：同一文件重复调用安全，已完成的直接跳过

使用示例
--------
    from core.tasks import IngestTask

    # 同步执行（直接调用，适合测试和简单场景）
    result = IngestTask.run("uploads/invoice.jpg")
    print(result)
    # {
    #   "status": "done",
    #   "file_id": "abc123...",
    #   "docs": 1,
    #   "chunks": 3,
    #   "new_chunks": 3,
    #   "vectors": 3,
    #   "elapsed": 8.2,
    # }

    # 跳过（文件已入库）
    result = IngestTask.run("uploads/invoice.jpg")
    # {"status": "skipped", "reason": "文件已入库"}

    # 异步执行（FastAPI BackgroundTasks）
    background_tasks.add_task(IngestTask.run, file_path)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from config.settings import settings
from core.dedup import Dedup
from core.embedder import Embedder
from extract_processor import ExtractProcessor
from extractor.doc_chunker import DocChunker

logger = logging.getLogger(__name__)


class IngestTask:
    """
    文件入库任务，所有方法均为类方法，无需实例化。
    """

    @classmethod
    def run(
        cls,
        file_path: str,
        *,
        force: bool = False,
        extract_kwargs: dict | None = None,
    ) -> dict:
        """
        执行完整入库流水线。

        Parameters
        ----------
        file_path      : 已保存到磁盘的文件路径
        force          : True = 忽略去重，强制重新处理
        extract_kwargs : 透传给 ExtractProcessor.extract() 的额外参数
                         （例如 OCR 后端配置，默认从 settings 读）

        Returns
        -------
        dict，包含 status / file_id / 各步骤数量 / elapsed
        """
        path = Path(file_path)
        start = time.perf_counter()

        logger.info("=" * 60)
        logger.info("[Task] 开始处理: %s", path.name)
        logger.info("[Task] 文件大小: %.2f MB", path.stat().st_size / 1024 / 1024)

        # ── STEP 0: 文件级去重 ─────────────────────────────────────────
        if not force and not Dedup.is_file_new(path):
            logger.info("[Task] 文件已入库，跳过: %s", path.name)
            return {"status": "skipped", "reason": "文件已入库", "file": str(path)}

        file_id = Dedup.register_file(path)
        logger.info("[Task] file_id: %s", file_id[:16])

        try:
            result = cls._run_pipeline(path, file_id, extract_kwargs or {})
        except Exception as exc:
            Dedup.mark_failed(file_id)
            logger.error("[Task] 处理失败: %s  错误: %s", path.name, exc, exc_info=True)
            return {
                "status":  "failed",
                "file_id": file_id,
                "file":    str(path),
                "error":   str(exc),
                "elapsed": round(time.perf_counter() - start, 2),
            }

        Dedup.mark_done(file_id)
        result["elapsed"] = round(time.perf_counter() - start, 2)

        logger.info(
            "[Task] 完成: %s  chunks=%d  new=%d  耗时=%.2fs",
            path.name, result["chunks"], result["new_chunks"], result["elapsed"],
        )
        logger.info("=" * 60)
        return result

    # ------------------------------------------------------------------
    #  内部流水线
    # ------------------------------------------------------------------

    @classmethod
    def _run_pipeline(
        cls,
        path: Path,
        file_id: str,
        extract_kwargs: dict,
    ) -> dict:

        # ── STEP 1: 解析 ───────────────────────────────────────────────
        logger.info("[Task] STEP 1 解析: %s", path.name)
        t0 = time.perf_counter()

        # 默认从 settings 读 OCR 配置，调用方可通过 extract_kwargs 覆盖
        ocr_defaults = {
            "vl_rec_backend":   settings.vl_backend,
            "vl_rec_server_url": settings.vl_server_url,
            "device":           settings.vl_device,
            "max_file_mb":      settings.vl_max_file_mb,
        }
        # 过滤掉 None 值，避免覆盖 ImageExtractor 的默认逻辑
        ocr_defaults = {k: v for k, v in ocr_defaults.items() if v is not None}
        ocr_defaults.update(extract_kwargs)

        docs = ExtractProcessor.extract(str(path), **ocr_defaults)
        logger.info(
            "[Task] 解析完成: %d 个 doc，耗时 %.2fs", len(docs), time.perf_counter() - t0
        )

        for i, doc in enumerate(docs):
            logger.debug(
                "[Task]   doc[%d] label=%s  字符=%d  内容: %r",
                i, doc.metadata.get("label", "?"),
                len(doc.page_content), doc.page_content[:80],
            )

        if not docs or all(not d.page_content.strip() for d in docs):
            raise ValueError(f"解析结果为空: {path.name}")

        # ── STEP 2: 切片 ───────────────────────────────────────────────
        logger.info("[Task] STEP 2 切片")
        t0 = time.perf_counter()

        chunks = DocChunker.chunk(docs, source_override=str(path))
        logger.info(
            "[Task] 切片完成: %d 个 chunk，耗时 %.2fs",
            len(chunks), time.perf_counter() - t0,
        )

        for i, chunk in enumerate(chunks):
            logger.debug(
                "[Task]   chunk[%d] strategy=%s  字符=%d  内容: %r",
                i, chunk.metadata.get("chunk_strategy", "?"),
                len(chunk.page_content), chunk.page_content[:60],
            )

        # ── STEP 3: Chunk 级去重 ───────────────────────────────────────
        logger.info("[Task] STEP 3 Chunk 去重")
        t0 = time.perf_counter()

        new_chunks, new_hashes = Dedup.filter_new_chunks(chunks, file_id)
        logger.info(
            "[Task] 去重完成: 共 %d，跳过 %d，新增 %d，耗时 %.2fs",
            len(chunks), len(chunks) - len(new_chunks),
            len(new_chunks), time.perf_counter() - t0,
        )

        if not new_chunks:
            logger.info("[Task] 所有 chunk 已入库，跳过向量化")
            return {
                "status":     "done",
                "file_id":    file_id,
                "file":       str(path),
                "docs":       len(docs),
                "chunks":     len(chunks),
                "new_chunks": 0,
                "vectors":    0,
            }

        # ── STEP 4: 向量化 ─────────────────────────────────────────────
        logger.info("[Task] STEP 4 向量化 %d 个 chunk", len(new_chunks))
        t0 = time.perf_counter()

        vectors = Embedder.embed(new_chunks)
        logger.info(
            "[Task] 向量化完成: %d 个向量，维度=%d，耗时 %.2fs",
            len(vectors), len(vectors[0]) if vectors else 0,
            time.perf_counter() - t0,
        )

        # ── STEP 5: 落库 ───────────────────────────────────────────────
        logger.info("[Task] STEP 5 Milvus 落库")
        t0 = time.perf_counter()

        written = Embedder.store(new_chunks, vectors, file_id, new_hashes)
        logger.info(
            "[Task] 落库完成: %d 条，耗时 %.2fs", written, time.perf_counter() - t0
        )

        # ── STEP 6: 登记 chunk 到 SQLite ──────────────────────────────
        Dedup.register_chunks([
            (h, file_id, chunk.metadata.get("chunk_index", i))
            for i, (chunk, h) in enumerate(zip(new_chunks, new_hashes))
        ])
        logger.debug("[Task] chunk 记录已写入 SQLite")

        return {
            "status":     "done",
            "file_id":    file_id,
            "file":       str(path),
            "docs":       len(docs),
            "chunks":     len(chunks),
            "new_chunks": len(new_chunks),
            "vectors":    written,
        }