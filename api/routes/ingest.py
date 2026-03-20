"""
api/routes/ingest.py — 文件上传入库接口。

接口
----
POST /api/ingest
    上传文件 → 保存磁盘 → 文件去重 → 解析 → 切片 → 向量化 → 入库

GET  /api/ingest/status/{file_id}
    查询入库状态

GET  /api/files
    已入库文件列表

DELETE /api/files/{file_id}
    删除文件及其向量
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from config.settings import settings
from core.dedup import Dedup
from core.embedder import Embedder
from core.tasks import IngestTask

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["ingest"])

# ---------------------------------------------------------------------------
# 支持的文件格式
# ---------------------------------------------------------------------------

_ALLOWED_EXTENSIONS = {
    # 图片
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif",
    # 文档
    ".pdf", ".docx", ".pptx",
    # 表格
    ".xlsx", ".xls", ".csv",
    # 文本
    ".txt", ".md", ".html", ".htm",
}

_MAX_UPLOAD_MB = 50  # 上传大小限制


# ---------------------------------------------------------------------------
# POST /api/ingest — 上传并入库
# ---------------------------------------------------------------------------

@router.post("/ingest")
async def ingest(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    sync: bool = False,   # True = 同步等待结果；False = 后台异步处理
):
    """
    上传文件并触发入库流水线。

    - **sync=false**（默认）：立即返回 file_id，后台异步处理
    - **sync=true**：等待处理完成后返回结果（适合调试和小文件）
    """
    # ── 文件格式校验 ──────────────────────────────────────────────────
    filename = file.filename or "unknown"
    ext = Path(filename).suffix.lower()
    if ext not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"不支持的文件格式: {ext}，支持: {', '.join(sorted(_ALLOWED_EXTENSIONS))}",
        )

    # ── 保存到磁盘 ────────────────────────────────────────────────────
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    save_path = upload_dir / filename
    # 文件名冲突时加序号
    counter = 1
    while save_path.exists():
        save_path = upload_dir / f"{Path(filename).stem}_{counter}{ext}"
        counter += 1

    content = await file.read()

    # 大小限制
    size_mb = len(content) / 1024 / 1024
    if size_mb > _MAX_UPLOAD_MB:
        raise HTTPException(
            status_code=413,
            detail=f"文件过大: {size_mb:.1f} MB，限制 {_MAX_UPLOAD_MB} MB",
        )

    save_path.write_bytes(content)
    logger.info("[Ingest] 文件保存: %s (%.2f MB)", save_path.name, size_mb)

    # ── 文件级去重（保存后立即检查）──────────────────────────────────
    file_id = Dedup.hash_file(save_path)
    if not Dedup.is_file_new(save_path):
        # 已入库，删除刚保存的重复文件
        save_path.unlink(missing_ok=True)
        existing = Dedup.get_file(file_id) or {}
        logger.info("[Ingest] 文件已存在，跳过: %s", file_id[:16])
        return JSONResponse({
            "status":    "skipped",
            "reason":    "文件内容已入库",
            "file_id":   file_id,
            "file":      existing.get("filename", filename),
        })

    # ── 触发入库流水线 ────────────────────────────────────────────────
    if sync:
        # 同步模式：等待完成
        result = IngestTask.run(str(save_path))
        return JSONResponse(result)
    else:
        # 异步模式：后台处理，立即返回
        background_tasks.add_task(IngestTask.run, str(save_path))
        return JSONResponse({
            "status":   "processing",
            "file_id":  file_id,
            "file":     save_path.name,
            "message":  "文件已上传，正在后台处理",
        }, status_code=202)


# ---------------------------------------------------------------------------
# GET /api/ingest/status/{file_id} — 查询状态
# ---------------------------------------------------------------------------

@router.get("/ingest/status/{file_id}")
async def ingest_status(file_id: str):
    """查询文件入库状态。"""
    record = Dedup.get_file(file_id)
    if not record:
        raise HTTPException(status_code=404, detail="文件不存在")

    resp = {
        "file_id":  record["id"],
        "filename": record["filename"],
        "status":   record["status"],   # pending / done / failed
        "created_at": record["created_at"],
    }

    # done 状态额外返回 chunk 数量
    if record["status"] == "done":
        try:
            resp["total_vectors"] = Embedder.count()
        except Exception:
            pass

    return JSONResponse(resp)


# ---------------------------------------------------------------------------
# GET /api/files — 已入库文件列表
# ---------------------------------------------------------------------------

@router.get("/files")
async def list_files(status: str | None = None):
    """
    列出已入库文件。

    - status=done    : 只看成功的
    - status=failed  : 只看失败的
    - 不传           : 全部
    """
    files = Dedup.list_files(status=status)
    return JSONResponse({
        "total": len(files),
        "files": [
            {
                "file_id":    f["id"][:16] + "...",
                "filename":   f["filename"],
                "status":     f["status"],
                "created_at": f["created_at"],
            }
            for f in files
        ],
    })


# ---------------------------------------------------------------------------
# DELETE /api/files/{file_id} — 删除文件
# ---------------------------------------------------------------------------

@router.delete("/files/{file_id}")
async def delete_file(file_id: str):
    """删除文件的向量数据和元数据记录（不删除磁盘原文件）。"""
    record = Dedup.get_file(file_id)
    if not record:
        raise HTTPException(status_code=404, detail="文件不存在")

    # 删除 Milvus 向量
    try:
        Embedder.delete_by_file(file_id)
    except Exception as e:
        logger.warning("[Ingest] Milvus 删除失败: %s", e)

    # 删除 SQLite 记录
    Dedup.delete_file(file_id)

    logger.info("[Ingest] 删除文件记录: %s (%s)", record["filename"], file_id[:16])
    return JSONResponse({
        "status":   "deleted",
        "file_id":  file_id,
        "filename": record["filename"],
    })