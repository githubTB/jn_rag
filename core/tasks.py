"""
core/tasks.py — 完整入库流水线。

串联：文件去重 → 解析 → 切片 → Chunk 去重 → 向量化 → 落库 → 标记完成

使用示例
--------
    from core.tasks import IngestTask

    # 单文件，带企业和类型
    result = IngestTask.run(
        "uploads/营业执照.jpg",
        task_id="task_001",
        doc_type="license",
    )

    # 同步调用（适合调试）
    result = IngestTask.run("uploads/合同.pdf", task_id="task_001", sync=True)

    # 异步（FastAPI BackgroundTasks）
    background_tasks.add_task(IngestTask.run, file_path,
                               task_id=task_id, doc_type=doc_type)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from config.settings import settings
from core.dedup import Dedup, DocType
from core.doc_type_classifier import classify_doc_type
from core.embedder import Embedder
from extract_processor import ExtractProcessor
from extractor.doc_chunker import DocChunker

logger = logging.getLogger(__name__)


_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}


def _should_auto_classify(path: Path, docs: list) -> bool:
    """
    业务 doc_type 自动细分只发生在两类特殊输入上：
    1. 原始图片文件
    2. 走过 OCR / 版面识别的 PDF
    其他固定格式（docx/pptx/html/text 等）不在这里重判业务类型。
    """
    suffix = path.suffix.lower()
    if suffix in _IMAGE_SUFFIXES:
        return True
    if suffix != ".pdf":
        return False
    labels = {str(doc.metadata.get("label", "")).lower() for doc in docs}
    return any(label and label != "page" for label in labels)


class IngestTask:
    """文件入库任务，所有方法均为类方法，无需实例化。"""

    @classmethod
    def run(
        cls,
        file_path: str,
        *,
        task_id: str | None = None,
        doc_type: str = DocType.UNKNOWN,
        doc_type_confirmed: bool = False,
        force: bool = False,
        extract_kwargs: dict | None = None,
    ) -> dict:
        """
        执行完整入库流水线。

        Parameters
        ----------
        file_path      : 已保存到磁盘的文件路径
        task_id        : 所属任务 ID（None 表示不关联任务）
        doc_type       : 文件类型，见 DocType 常量
        force          : True = 忽略去重，强制重新处理
        extract_kwargs : 透传给 ExtractProcessor.extract() 的额外参数

        Returns
        -------
        dict，含 status / file_id / task_id / doc_type / 各步骤数量 / elapsed
        """
        path  = Path(file_path)
        start = time.perf_counter()

        logger.info("=" * 60)
        logger.info("[Task] 开始: %s  task=%s  type=%s",
                    path.name, task_id, doc_type)
        logger.info("[Task] 文件大小: %.2f MB", path.stat().st_size / 1024 / 1024)

        # ── STEP 0: 文件级去重 ─────────────────────────────────────────
        if not force and not Dedup.is_file_new(path):
            logger.info("[Task] 文件已入库，跳过: %s", path.name)
            return {
                "status":     "skipped",
                "reason":     "文件已入库",
                "file":       str(path),
                "task_id":    task_id,
                "doc_type":   doc_type,
            }

        file_id = Dedup.register_file(
            path,
            task_id=task_id,
            doc_type=doc_type,
            doc_type_confirmed=doc_type_confirmed,
        )
        logger.info("[Task] file_id: %s", file_id[:16])

        try:
            result = cls._run_pipeline(
                path, file_id, task_id, doc_type, extract_kwargs or {}
            )
        except Exception as exc:
            Dedup.mark_failed(file_id)
            logger.error("[Task] 失败: %s  %s", path.name, exc, exc_info=True)
            return {
                "status":     "failed",
                "file_id":    file_id,
                "file":       str(path),
                "task_id":    task_id,
                "doc_type":   doc_type,
                "error":      str(exc),
                "elapsed":    round(time.perf_counter() - start, 2),
            }

        Dedup.mark_done(file_id)
        result["elapsed"] = round(time.perf_counter() - start, 2)
        logger.info(
            "[Task] 完成: %s  chunks=%d  new=%d  %.2fs",
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
        task_id: str | None,
        doc_type: str,
        extract_kwargs: dict,
    ) -> dict:

        # ── STEP 1: 解析 ───────────────────────────────────────────────
        logger.info("[Task] STEP 1 解析")
        t0 = time.perf_counter()

        ocr_defaults = {
            "vl_backend":        settings.vl_backend,
            "vl_base_url":       settings.vl_base_url,
            "device":            settings.vl_device,
            "max_file_mb":       settings.vl_max_file_mb,
            "doc_type":          doc_type,
        }
        ocr_defaults = {k: v for k, v in ocr_defaults.items() if v is not None}
        ocr_defaults.update(extract_kwargs)

        docs = ExtractProcessor.extract(str(path), **ocr_defaults)
        logger.info("[Task] 解析完成: %d doc，%.2fs", len(docs), time.perf_counter() - t0)
        
        record = Dedup.get_file(file_id) or {}
        auto_classify = not bool(record.get("doc_type_confirmed", 0))
        if auto_classify and docs and _should_auto_classify(path, docs):
            decision = classify_doc_type(docs, file_path=path)
            inferred = docs[0].metadata.get("inferred_doc_type")
            candidate = decision.doc_type
            if inferred in DocType.ALL and inferred != DocType.UNKNOWN:
                candidate = inferred
            if candidate in DocType.ALL and candidate != doc_type:
                logger.info(
                    "[Task] 内容分类更新文档类型: %s %s → %s (confidence=%.2f evidence=%s)",
                    path.name,
                    doc_type,
                    candidate,
                    decision.confidence,
                    ",".join(decision.evidence),
                )
                doc_type = candidate
                Dedup.update_doc_type(file_id, doc_type, confirmed=False)
    
        for i, doc in enumerate(docs):
            logger.debug("[Task]   doc[%d] label=%s 字符=%d 内容: %r",
                         i, doc.metadata.get("label","?"),
                         len(doc.page_content), doc.page_content[:80])

        if not docs or all(not d.page_content.strip() for d in docs):
            raise ValueError(f"解析结果为空: {path.name}")

        # ── STEP 2: 切片 ───────────────────────────────────────────────
        logger.info("[Task] STEP 2 切片")
        t0 = time.perf_counter()

        chunks = DocChunker.chunk(docs, source_override=str(path))
        logger.info("[Task] 切片完成: %d chunk，%.2fs", len(chunks), time.perf_counter() - t0)

        for i, chunk in enumerate(chunks):
            logger.debug("[Task]   chunk[%d] strategy=%s 字符=%d 内容: %r",
                         i, chunk.metadata.get("chunk_strategy","?"),
                         len(chunk.page_content), chunk.page_content[:60])

        # ── STEP 3: Chunk 去重 ─────────────────────────────────────────
        logger.info("[Task] STEP 3 Chunk 去重")
        t0 = time.perf_counter()

        new_chunks, new_hashes = Dedup.filter_new_chunks(chunks, file_id)
        logger.info("[Task] 去重: 共 %d，跳过 %d，新增 %d，%.2fs",
                    len(chunks), len(chunks) - len(new_chunks),
                    len(new_chunks), time.perf_counter() - t0)

        if not new_chunks:
            logger.info("[Task] 所有 chunk 已入库，跳过向量化")
            return {
                "status":     "done",
                "file_id":    file_id,
                "file":       str(path),
                "task_id":    task_id,
                "doc_type":   doc_type,
                "docs":       len(docs),
                "chunks":     len(chunks),
                "new_chunks": 0,
                "vectors":    0,
            }

        # ── STEP 4: 向量化 ─────────────────────────────────────────────
        logger.info("[Task] STEP 4 向量化 %d chunk", len(new_chunks))
        t0 = time.perf_counter()

        vectors = Embedder.embed(new_chunks)
        logger.info("[Task] 向量化完成: %d 向量，维度=%d，%.2fs",
                    len(vectors), len(vectors[0]) if vectors else 0,
                    time.perf_counter() - t0)

        # ── STEP 5: 落库 ───────────────────────────────────────────────
        logger.info("[Task] STEP 5 Milvus 落库")
        t0 = time.perf_counter()

        written = Embedder.store(
            new_chunks, vectors, file_id, new_hashes,
            task_id=task_id,
            doc_type=doc_type,
        )
        logger.info("[Task] 落库: %d 条，%.2fs", written, time.perf_counter() - t0)

        # ── STEP 6: 登记 chunk 到 SQLite ──────────────────────────────
        Dedup.register_chunks([
            (h, file_id, chunk.metadata.get("chunk_index", i))
            for i, (chunk, h) in enumerate(zip(new_chunks, new_hashes))
        ])
        logger.debug("[Task] chunk 记录写入 SQLite 完成")

        return {
            "status":     "done",
            "file_id":    file_id,
            "file":       str(path),
            "task_id":    task_id,
            "doc_type":   doc_type,
            "docs":       len(docs),
            "chunks":     len(chunks),
            "new_chunks": len(new_chunks),
            "vectors":    written,
        }
