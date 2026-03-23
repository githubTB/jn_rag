"""
core/tasks_celery.py — Celery 异步任务定义。

把原来的 IngestTask.run() 包装成 Celery 任务，
Worker 进程独立运行，不阻塞 FastAPI 主进程。
"""

from __future__ import annotations

import logging
from pathlib import Path

from celery_app import app
from core.tasks import IngestTask

logger = logging.getLogger(__name__)


@app.task(
    bind=True,
    name="core.tasks_celery.ingest_file",
    max_retries=3,
    default_retry_delay=10,
)
def ingest_file(
    self,
    file_path: str,
    company_id: str | None = None,
    doc_type: str = "unknown",
    force: bool = False,
) -> dict:
    """
    单文件入库任务。

    Parameters
    ----------
    file_path  : 文件路径
    company_id : 企业 ID
    doc_type   : 文件类型
    force      : 是否强制重新处理

    Returns
    -------
    dict，含 status / file_id / chunks / elapsed 等
    """
    logger.info("[CeleryTask] 开始: %s  company=%s  type=%s",
                Path(file_path).name, company_id, doc_type)
    try:
        result = IngestTask.run(
            file_path,
            company_id=company_id,
            doc_type=doc_type,
            force=force,
        )
        logger.info("[CeleryTask] 完成: %s  status=%s", Path(file_path).name, result.get("status"))
        return result
    except Exception as exc:
        logger.error("[CeleryTask] 失败: %s  %s", Path(file_path).name, exc, exc_info=True)
        # 自动重试
        raise self.retry(exc=exc)


@app.task(
    bind=True,
    name="core.tasks_celery.ingest_batch",
    max_retries=1,
)
def ingest_batch(
    self,
    tasks: list[dict],
    company_id: str,
) -> dict:
    """
    批量入库任务（扫描目录后用）。
    串行处理每个文件，保证顺序。

    Parameters
    ----------
    tasks      : [{"path": ..., "doc_type": ...}, ...]
    company_id : 企业 ID

    Returns
    -------
    dict，含每个文件的处理结果
    """
    total   = len(tasks)
    results = []
    ok = fail = skip = 0

    logger.info("[CeleryBatch] 开始批量入库: company=%s  共 %d 个文件", company_id, total)

    for i, task in enumerate(tasks, 1):
        file_path = task["path"]
        doc_type  = task.get("doc_type", "unknown")
        fname     = Path(file_path).name

        logger.info("[CeleryBatch] %d/%d  %s", i, total, fname)

        try:
            result = IngestTask.run(
                file_path,
                company_id=company_id,
                doc_type=doc_type,
            )
            status = result.get("status", "unknown")
            results.append({"file": fname, "doc_type": doc_type,
                             "status": status, "chunks": result.get("new_chunks", 0)})
            if status == "done":    ok += 1
            elif status == "skipped": skip += 1
            else: fail += 1
        except Exception as exc:
            logger.error("[CeleryBatch] 文件失败: %s  %s", fname, exc)
            results.append({"file": fname, "doc_type": doc_type,
                             "status": "failed", "error": str(exc)})
            fail += 1

    logger.info("[CeleryBatch] 完成: ok=%d  skip=%d  fail=%d", ok, skip, fail)

    return {
        "status":     "done",
        "company_id": company_id,
        "total":      total,
        "ok":         ok,
        "skipped":    skip,
        "failed":     fail,
        "results":    results,
    }