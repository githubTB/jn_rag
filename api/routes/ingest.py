"""
api/routes/ingest.py 里 scan 接口的改动部分。

把原来的 BackgroundTasks 替换成 Celery 任务。
其他接口（upload/folder、单文件入库、企业管理等）不变。

只需替换 scan_company_dir 函数。
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from config.settings import settings
from core.dedup import Dedup, DocType
from core.embedder import Embedder
from core.tasks import IngestTask

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["ingest"])

_ALLOWED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif",
    ".pdf", ".docx", ".pptx",
    ".xlsx", ".xls", ".csv",
    ".txt", ".md", ".html", ".htm",
}
_MAX_UPLOAD_MB = 50


# ---------------------------------------------------------------------------
# POST /api/ingest/scan/{company_id} — 扫描目录，用 Celery 异步入库
# ---------------------------------------------------------------------------

@router.post("/ingest/scan/{company_id}")
async def scan_company_dir(
    company_id: str,
    doc_type: str | None = None,
    sync: bool = False,
):
    """
    扫描企业目录下所有文件并入库。

    sync=false（默认）：把任务推入 Celery 队列，立即返回 task_id
    sync=true         ：等待 Celery 任务完成再返回（调试用，文件多时慢）
    """
    if not Dedup.get_company(company_id):
        raise HTTPException(status_code=404, detail=f"企业不存在: {company_id}")

    scan_dir = Path(settings.upload_dir) / company_id
    if not scan_dir.exists():
        raise HTTPException(status_code=404, detail=f"目录不存在: {scan_dir}")

    # 收集文件
    all_files: list[Path] = []
    for f in scan_dir.rglob("*"):
        if not f.is_file():
            continue
        if f.name in {".DS_Store", "Thumbs.db", ".gitkeep"}:
            continue
        if f.suffix.lower() in {".dwg", ".dxf", ".ds_store"}:
            continue
        prepared = _prepare_doc(f)
        if prepared is None:
            continue
        supported = _ALLOWED_EXTENSIONS | {".doc", ".docm"}
        if f.suffix.lower() not in supported:
            continue
        all_files.append(prepared)

    if not all_files:
        return JSONResponse({"status": "empty", "company_id": company_id,
                             "message": "目录下没有找到支持的文件"})

    tasks = [
        {"path": str(f), "doc_type": doc_type or _guess_doc_type(f)}
        for f in all_files
    ]

    # 导入 Celery 任务
    from core.tasks_celery import ingest_batch

    if sync:
        # 同步等待（调试用）
        result = ingest_batch.apply(args=[tasks, company_id]).get(timeout=3600)
        return JSONResponse(result)
    else:
        # 异步推队列，立即返回 task_id
        job = ingest_batch.apply_async(args=[tasks, company_id], queue="ingest")
        logger.info("[Scan] 已推入队列: company=%s  files=%d  task_id=%s",
                    company_id, len(tasks), job.id)
        return JSONResponse({
            "status":    "queued",
            "task_id":   job.id,
            "company_id": company_id,
            "total":     len(tasks),
            "files":     [{"file": Path(t["path"]).name, "doc_type": t["doc_type"]} for t in tasks],
            "message":   f"已提交 {len(tasks)} 个文件到队列，Worker 正在处理",
        }, status_code=202)


# ---------------------------------------------------------------------------
# GET /api/task/{task_id} — 查询 Celery 任务状态
# ---------------------------------------------------------------------------

@router.get("/task/{task_id}")
async def get_task_status(task_id: str):
    """查询任务处理进度。"""
    from celery_app import app as celery_app
    from celery.result import AsyncResult

    result = AsyncResult(task_id, app=celery_app)

    resp = {
        "task_id": task_id,
        "status":  result.status,   # PENDING / STARTED / SUCCESS / FAILURE / RETRY
    }

    if result.successful():
        resp["result"] = result.result
    elif result.failed():
        resp["error"] = str(result.result)

    return JSONResponse(resp)


# ---------------------------------------------------------------------------
# 工具函数（保持不变）
# ---------------------------------------------------------------------------

_PATH_RULES: list[tuple[list[str], str]] = [
    (["营业执照", "执照", "license", "证照", "信用中国"],                      "license"),
    (["排污许可", "许可证"],                                                    "license"),
    (["现场照片", "铭牌", "nameplate", "设备铭牌"],                             "nameplate"),
    (["微信图片", "现场"],                                                       "nameplate"),
    (["统计表", "汇总", "台账", "明细", "一览表", "固废处理统计", "废水废气"], "table"),
    (["水电气", "生产设备"],                                                     "table"),
    (["检测报告", "监测报告", "废气", "废水", "噪音", "环境影响", "环评",
      "排污", "验收", "竣工", "锅炉", "分析报告"],                             "document"),
    (["认证", "证书", "资质", "许可", "批准书"],                                "document"),
    (["合同", "协议", "制度", "规程", "工艺"],                                   "document"),
    (["总平面图", "平面图", "报告书", "报告"],                                    "document"),
    (["收资清单", "评估报告", "诊断报告", "GMP", "AL-SMP"],                     "document"),
    (["危废", "固废", "废品", "污水"],                                           "document"),
    (["发票", "invoice", "单据", "收据"],                                        "invoice"),
]


def _guess_doc_type(file_path: Path) -> str:
    path_str = "/".join(file_path.parts).lower()
    for keywords, dt in _PATH_RULES:
        if any(k.lower() in path_str for k in keywords):
            return dt
    suffix = file_path.suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}:
        return "nameplate"
    return "document"


def _prepare_doc(file_path: Path) -> Path | None:
    suffix = file_path.suffix.lower()
    if suffix in {".dwg", ".dxf", ".ds_store"}:
        return None
    if suffix == ".doc":
        if file_path.with_suffix(".docx").exists():
            return None
        return file_path
    return file_path