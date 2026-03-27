"""
api/routes/ingest.py — 完整版，含所有接口。

接口列表
--------
POST   /api/upload/folder                  文件夹批量上传（保持目录结构，不触发入库）
POST   /api/ingest                         单文件上传入库
POST   /api/ingest/scan/{company_id}       扫描目录批量入库（Celery 异步队列）
GET    /api/ingest/status/{file_id}        查询入库状态
GET    /api/task/{task_id}                 查询 Celery 任务状态
GET    /api/files                          已入库文件列表
GET    /api/files/{file_id}/content        查看文件完整解析原文
GET    /api/files/{file_id}/asset          下载或浏览器预览原文件
DELETE /api/files/{file_id}                删除文件
PATCH  /api/files/{file_id}/doc_type       人工修改文件类型
POST   /api/files/{file_id}/replace        覆盖上传并重新入库
POST   /api/files/{file_id}/reprocess      重新处理文件
POST   /api/files/{file_id}/reclassify     按已入库内容重新分类

POST   /api/companies                      创建企业
GET    /api/companies                      企业列表
GET    /api/companies/{company_id}         企业详情
DELETE /api/companies/{company_id}         删除企业
"""

from __future__ import annotations

import logging
import mimetypes
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from config.settings import settings
from core.dedup import Dedup, DocType
from core.doc_type_classifier import classify_doc_type
from core.embedder import Embedder
from core.tasks import IngestTask
from models.document import Document

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
# POST /api/upload/folder — 文件夹批量上传，保持目录结构，不触发入库
# ---------------------------------------------------------------------------

@router.post("/upload/folder")
async def upload_folder(
    files: list[UploadFile] = File(...),
    relative_paths: list[str] = Form(...),
    company_id: str = Form(...),
    company_name: str = Form(...),
):
    """
    上传整个文件夹，保持目录结构，只保存不入库。
    入库请调用 POST /api/ingest/scan/{company_id}
    """
    if len(files) != len(relative_paths):
        raise HTTPException(status_code=400, detail="files 和 relative_paths 数量不一致")

    Dedup.register_company(company_id, company_name)
    logger.info("[Upload] 企业: %s (%s)  文件数: %d", company_name, company_id, len(files))

    base_dir = Path(settings.upload_dir) / company_id
    saved, skipped_list = [], []

    for file, rel_path in zip(files, relative_paths):
        parts = Path(rel_path).parts
        safe_rel = Path(*parts[1:]) if len(parts) > 1 else Path(parts[0])

        ext = safe_rel.suffix.lower()
        if ext not in _ALLOWED_EXTENSIONS:
            skipped_list.append(str(safe_rel))
            continue

        content = await file.read()
        size_mb = len(content) / 1024 / 1024
        if size_mb > _MAX_UPLOAD_MB:
            skipped_list.append(str(safe_rel))
            logger.warning("[Upload] 文件过大跳过: %s (%.1fMB)", safe_rel, size_mb)
            continue

        save_path = base_dir / safe_rel
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_bytes(content)
        saved.append({"file": str(safe_rel), "size_mb": round(size_mb, 2)})

    logger.info("[Upload] 完成: 保存 %d 个，跳过 %d 个", len(saved), len(skipped_list))

    return JSONResponse({
        "status":        "uploaded",
        "company_id":    company_id,
        "company_name":  company_name,
        "saved":         len(saved),
        "skipped":       len(skipped_list),
        "files":         saved,
        "skipped_files": skipped_list,
        "message":       f"上传完成，共 {len(saved)} 个文件。点击「开始入库」处理。",
    })


# ---------------------------------------------------------------------------
# POST /api/ingest — 单文件上传入库
# ---------------------------------------------------------------------------

@router.post("/ingest")
async def ingest(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    company_id: str | None = None,
    doc_type: str | None = None,
    sync: bool = False,
):
    if company_id and not Dedup.get_company(company_id):
        raise HTTPException(status_code=404,
            detail=f"企业不存在: {company_id}，请先通过 POST /api/companies 创建")

    doc_type_confirmed = doc_type is not None
    if doc_type is None:
        doc_type = _guess_doc_type(Path(file.filename or "unknown"))
    elif doc_type not in DocType.ALL:
        raise HTTPException(status_code=400,
            detail=f"无效的 doc_type: {doc_type}，可选: {', '.join(sorted(DocType.ALL))}")

    filename = file.filename or "unknown"
    ext = Path(filename).suffix.lower()
    if ext not in _ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=415, detail=f"不支持的文件格式: {ext}")

    upload_dir = Path(settings.upload_dir)
    if company_id:
        upload_dir = upload_dir / company_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    save_path = upload_dir / filename
    counter = 1
    while save_path.exists():
        save_path = upload_dir / f"{Path(filename).stem}_{counter}{ext}"
        counter += 1

    content = await file.read()
    size_mb = len(content) / 1024 / 1024
    if size_mb > _MAX_UPLOAD_MB:
        raise HTTPException(status_code=413, detail=f"文件过大: {size_mb:.1f} MB")
    save_path.write_bytes(content)

    file_id = Dedup.hash_file(save_path)
    if not Dedup.is_file_new(save_path):
        save_path.unlink(missing_ok=True)
        existing = Dedup.get_file(file_id) or {}
        return JSONResponse({"status": "skipped", "reason": "文件内容已入库",
                             "file_id": file_id, "file": existing.get("filename", filename)})

    kwargs = dict(
        company_id=company_id,
        doc_type=doc_type,
        doc_type_confirmed=doc_type_confirmed,
    )
    if sync:
        result = IngestTask.run(str(save_path), **kwargs)
        return JSONResponse(result)
    else:
        background_tasks.add_task(IngestTask.run, str(save_path), **kwargs)
        return JSONResponse({"status": "processing", "file_id": file_id,
                             "file": save_path.name, "message": "文件已上传，正在后台处理"},
                            status_code=202)


# ---------------------------------------------------------------------------
# POST /api/ingest/scan/{company_id} — 扫描目录，Celery 异步队列
# ---------------------------------------------------------------------------

@router.post("/ingest/scan/{company_id}")
async def scan_company_dir(
    company_id: str,
    doc_type: str | None = None,
    sync: bool = False,
):
    if not Dedup.get_company(company_id):
        raise HTTPException(status_code=404, detail=f"企业不存在: {company_id}")

    scan_dir = Path(settings.upload_dir) / company_id
    if not scan_dir.exists():
        raise HTTPException(status_code=404, detail=f"目录不存在: {scan_dir}")

    all_files: list[Path] = []
    skipped_existing: list[str] = []
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
        if f.suffix.lower() not in (_ALLOWED_EXTENSIONS | {".doc", ".docm"}):
            continue
        if not Dedup.is_file_new(prepared):
            skipped_existing.append(prepared.name)
            continue
        all_files.append(prepared)

    if not all_files:
        message = "目录下没有找到需要入库的新文件" if skipped_existing else "目录下没有找到支持的文件"
        return JSONResponse({
            "status": "empty",
            "company_id": company_id,
            "message": message,
            "skipped_existing": len(skipped_existing),
        })

    tasks = [{
        "path": str(f),
        "doc_type": doc_type or _guess_doc_type(f),
        "doc_type_confirmed": doc_type is not None,
    }
             for f in all_files]

    from core.tasks_celery import ingest_batch

    if sync:
        result = ingest_batch.apply(args=[tasks, company_id]).get(timeout=3600)
        return JSONResponse(result)
    else:
        job = ingest_batch.apply_async(args=[tasks, company_id], queue="ingest")
        logger.info("[Scan] 推入队列: company=%s  files=%d  task_id=%s",
                    company_id, len(tasks), job.id)
        return JSONResponse({
            "status":     "queued",
            "task_id":    job.id,
            "company_id": company_id,
            "total":      len(tasks),
            "skipped_existing": len(skipped_existing),
            "files":      [{"file": Path(t["path"]).name, "doc_type": t["doc_type"]} for t in tasks],
            "message":    f"已提交 {len(tasks)} 个新文件到队列，跳过 {len(skipped_existing)} 个已入库文件",
        }, status_code=202)


# ---------------------------------------------------------------------------
# GET /api/ingest/status/{file_id}
# ---------------------------------------------------------------------------

@router.get("/ingest/status/{file_id}")
async def ingest_status(file_id: str):
    record = Dedup.get_file(file_id)
    if not record:
        raise HTTPException(status_code=404, detail="文件不存在")
    resp = {
        "file_id":            record["id"],
        "filename":           record["filename"],
        "company_id":         record["company_id"],
        "doc_type":           record["doc_type"],
        "doc_type_confirmed": bool(record.get("doc_type_confirmed", 0)),
        "status":             record["status"],
        "created_at":         record["created_at"],
    }
    if record["status"] == "done":
        try:
            resp["total_vectors"] = Embedder.count()
        except Exception:
            pass
    return JSONResponse(resp)


# ---------------------------------------------------------------------------
# GET /api/task/{task_id} — 查询 Celery 任务状态
# ---------------------------------------------------------------------------

@router.get("/task/{task_id}")
async def get_task_status(task_id: str):
    from celery_app import app as celery_app
    from celery.result import AsyncResult

    result = AsyncResult(task_id, app=celery_app)
    resp = {"task_id": task_id, "status": result.status}
    if result.successful():
        resp["result"] = result.result
    elif result.failed():
        resp["error"] = str(result.result)
    return JSONResponse(resp)


# ---------------------------------------------------------------------------
# GET /api/files
# ---------------------------------------------------------------------------

@router.get("/files")
async def list_files(
    status: str | None = None,
    company_id: str | None = None,
    doc_type: str | None = None,
):
    files = Dedup.list_files(status=status, company_id=company_id, doc_type=doc_type)
    return JSONResponse({
        "total": len(files),
        "files": [{
            "file_id":            f["id"][:16] + "..",
            "filename":           f["filename"],
            "company_id":         f["company_id"],
            "doc_type":           f["doc_type"],
            "doc_type_confirmed": bool(f.get("doc_type_confirmed", 0)),
            "status":             f["status"],
            "created_at":         f["created_at"],
        } for f in files],
    })


@router.get("/files/{file_id}/content")
async def get_file_content(file_id: str):
    record = Dedup.get_file(file_id)
    if not record:
        all_files = Dedup.list_files()
        matched = [f for f in all_files if f["id"].startswith(file_id)]
        if len(matched) == 1:
            record = matched[0]
            file_id = record["id"]
        elif len(matched) > 1:
            raise HTTPException(status_code=400, detail="file_id 前缀匹配到多个文件")
        else:
            raise HTTPException(status_code=404, detail="文件不存在")

    chunks = _query_file_chunks(file_id, limit=512)
    full_text = _merge_chunk_text(chunks, max_chars=None)

    return JSONResponse({
        "file_id": file_id,
        "filename": record["filename"],
        "doc_type": record["doc_type"],
        "status": record["status"],
        "full_text": full_text,
        "chunk_count": len(chunks),
    })


@router.get("/files/{file_id}/asset")
async def get_file_asset(
    file_id: str,
    disposition: str = Query("inline", pattern="^(inline|attachment)$"),
):
    record = Dedup.get_file(file_id)
    if not record:
        all_files = Dedup.list_files()
        matched = [f for f in all_files if f["id"].startswith(file_id)]
        if len(matched) == 1:
            record = matched[0]
            file_id = record["id"]
        elif len(matched) > 1:
            raise HTTPException(status_code=400, detail="file_id 前缀匹配到多个文件")
        else:
            raise HTTPException(status_code=404, detail="文件不存在")

    file_path = Path(record["file_path"])
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"原始文件不存在: {file_path}")

    media_type, _ = mimetypes.guess_type(str(file_path))
    return FileResponse(
        path=str(file_path),
        filename=record["filename"],
        media_type=media_type or "application/octet-stream",
        content_disposition_type=disposition,
    )


# ---------------------------------------------------------------------------
# DELETE /api/files/{file_id}
# ---------------------------------------------------------------------------

@router.delete("/files/{file_id}")
async def delete_file(file_id: str):
    record = Dedup.get_file(file_id)
    if not record:
        raise HTTPException(status_code=404, detail="文件不存在")
    try:
        Embedder.delete_by_file(file_id)
    except Exception as e:
        logger.warning("[Ingest] Milvus 删除失败: %s", e)
    Dedup.delete_file(file_id)
    return JSONResponse({"status": "deleted", "file_id": file_id,
                         "filename": record["filename"]})


# ---------------------------------------------------------------------------
# PATCH /api/files/{file_id}/doc_type
# ---------------------------------------------------------------------------

class DocTypeUpdate(BaseModel):
    doc_type: str
    confirmed: bool = True


@router.patch("/files/{file_id}/doc_type")
async def update_doc_type(file_id: str, body: DocTypeUpdate):
    if body.doc_type not in DocType.ALL:
        raise HTTPException(status_code=400,
            detail=f"无效的 doc_type: {body.doc_type}，可选: {', '.join(sorted(DocType.ALL))}")

    record = Dedup.get_file(file_id)
    if not record:
        all_files = Dedup.list_files()
        matched = [f for f in all_files if f["id"].startswith(file_id)]
        if len(matched) == 1:
            record = matched[0]; file_id = record["id"]
        elif len(matched) > 1:
            raise HTTPException(status_code=400, detail="file_id 前缀匹配到多个文件")
        else:
            raise HTTPException(status_code=404, detail="文件不存在")

    old_type = record["doc_type"]
    if not Dedup.update_doc_type(file_id, body.doc_type, confirmed=body.confirmed):
        raise HTTPException(status_code=404, detail="文件不存在")

    return JSONResponse({
        "status":             "updated",
        "file_id":            file_id,
        "filename":           record["filename"],
        "doc_type_old":       old_type,
        "doc_type_new":       body.doc_type,
        "doc_type_confirmed": body.confirmed,
        "message":            "doc_type 已更新，如需重新解析请调用 reprocess 接口",
    })


# ---------------------------------------------------------------------------
# POST /api/files/{file_id}/replace
# ---------------------------------------------------------------------------

@router.post("/files/{file_id}/replace")
async def replace_file(
    file_id: str,
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    sync: bool = False,
):
    record = Dedup.get_file(file_id)
    if not record:
        all_files = Dedup.list_files()
        matched = [f for f in all_files if f["id"].startswith(file_id)]
        if len(matched) == 1:
            record = matched[0]
            file_id = record["id"]
        elif len(matched) > 1:
            raise HTTPException(status_code=400, detail="file_id 前缀匹配到多个文件")
        else:
            raise HTTPException(status_code=404, detail="文件不存在")

    target_path = Path(record["file_path"])
    current_ext = target_path.suffix.lower()
    incoming_name = file.filename or record["filename"]
    incoming_ext = Path(incoming_name).suffix.lower()
    if current_ext and incoming_ext and current_ext != incoming_ext:
        raise HTTPException(status_code=400, detail=f"覆盖文件后缀不一致: {incoming_ext} != {current_ext}")

    content = await file.read()
    size_mb = len(content) / 1024 / 1024
    if size_mb > _MAX_UPLOAD_MB:
        raise HTTPException(status_code=413, detail=f"文件过大: {size_mb:.1f} MB")

    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_bytes(content)

    try:
        Embedder.delete_by_file(file_id)
    except Exception as exc:
        logger.warning("[Replace] Milvus 删除失败: %s", exc)
    Dedup.delete_file(file_id)

    kwargs = dict(
        company_id=record["company_id"],
        doc_type=record["doc_type"],
        doc_type_confirmed=bool(record.get("doc_type_confirmed", 0)),
        force=True,
    )

    if sync:
        result = IngestTask.run(str(target_path), **kwargs)
        return JSONResponse({
            **result,
            "replaced": True,
            "old_file_id": file_id,
            "filename": record["filename"],
        })

    background_tasks.add_task(IngestTask.run, str(target_path), **kwargs)
    return JSONResponse({
        "status": "processing",
        "replaced": True,
        "old_file_id": file_id,
        "filename": record["filename"],
        "message": "源文件已覆盖，正在重新入库",
    }, status_code=202)


# ---------------------------------------------------------------------------
# POST /api/files/{file_id}/reprocess
# ---------------------------------------------------------------------------

@router.post("/files/{file_id}/reprocess")
async def reprocess_file(
    file_id: str,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    sync: bool = False,
):
    record = Dedup.get_file(file_id)
    if not record:
        all_files = Dedup.list_files()
        matched = [f for f in all_files if f["id"].startswith(file_id)]
        if len(matched) == 1:
            record = matched[0]; file_id = record["id"]
        elif len(matched) > 1:
            raise HTTPException(status_code=400, detail="file_id 前缀匹配到多个文件")
        else:
            raise HTTPException(status_code=404, detail="文件不存在")

    file_path = Path(record["file_path"])
    if not file_path.exists():
        raise HTTPException(status_code=404,
            detail=f"原始文件不存在: {file_path}，无法重新处理")

    try:
        Embedder.delete_by_file(file_id)
    except Exception as e:
        logger.warning("[Reprocess] Milvus 删除失败: %s", e)

    deleted_chunks = Dedup.delete_chunks_by_file(file_id)

    with Dedup._get_conn_ctx() as conn:
        conn.execute("UPDATE files SET status = 'pending' WHERE id = ?", (file_id,))

    kwargs = dict(
        company_id=record["company_id"],
        doc_type=record["doc_type"],
        doc_type_confirmed=bool(record.get("doc_type_confirmed", 0)),
        force=True,
    )

    if sync:
        result = IngestTask.run(str(file_path), **kwargs)
        return JSONResponse({**result, "reprocessed": True,
                             "deleted_chunks": deleted_chunks})
    else:
        background_tasks.add_task(IngestTask.run, str(file_path), **kwargs)
        return JSONResponse({
            "status":         "processing",
            "file_id":        file_id,
            "filename":       record["filename"],
            "doc_type":       record["doc_type"],
            "deleted_chunks": deleted_chunks,
            "reprocessed":    True,
            "message":        "已清除旧数据，正在后台重新处理",
        }, status_code=202)


@router.post("/files/{file_id}/reclassify")
async def reclassify_file(file_id: str):
    record = Dedup.get_file(file_id)
    if not record:
        all_files = Dedup.list_files()
        matched = [f for f in all_files if f["id"].startswith(file_id)]
        if len(matched) == 1:
            record = matched[0]
            file_id = record["id"]
        elif len(matched) > 1:
            raise HTTPException(status_code=400, detail="file_id 前缀匹配到多个文件")
        else:
            raise HTTPException(status_code=404, detail="文件不存在")

    if bool(record.get("doc_type_confirmed", 0)):
        return JSONResponse({
            "status": "skipped",
            "file_id": file_id,
            "filename": record["filename"],
            "doc_type": record["doc_type"],
            "reason": "文件类型已人工确认，未自动覆盖",
        })

    docs = _load_docs_for_reclassify(record)
    if not docs:
        raise HTTPException(status_code=400, detail="无可用于重分类的解析内容，请先重新处理文件")

    decision = classify_doc_type(docs, file_path=record["file_path"])
    old_type = record["doc_type"]
    new_type = decision.doc_type if decision.doc_type in DocType.ALL else DocType.UNKNOWN
    Dedup.update_doc_type(file_id, new_type, confirmed=False)

    return JSONResponse({
        "status": "updated",
        "file_id": file_id,
        "filename": record["filename"],
        "doc_type_old": old_type,
        "doc_type_new": new_type,
        "doc_type_confirmed": False,
        "confidence": decision.confidence,
        "evidence": decision.evidence,
    })


# ---------------------------------------------------------------------------
# 企业管理
# ---------------------------------------------------------------------------

class CompanyCreate(BaseModel):
    company_id: str
    name: str


@router.post("/companies", status_code=201)
async def create_company(body: CompanyCreate):
    Dedup.register_company(body.company_id, body.name)
    return JSONResponse({"status": "ok", "company_id": body.company_id,
                         "name": body.name}, status_code=201)


@router.get("/companies")
async def list_companies():
    companies = Dedup.list_companies()
    result = []
    for c in companies:
        files = Dedup.list_files(company_id=c["id"])
        result.append({
            "company_id":      c["id"],
            "name":            c["name"],
            "created_at":      c["created_at"],
            "file_count":      len(files),
            "done_count":      sum(1 for f in files if f["status"] == "done"),
            "confirmed_count": sum(1 for f in files if f.get("doc_type_confirmed")),
        })
    return JSONResponse({"total": len(result), "companies": result})


@router.get("/companies/{company_id}")
async def get_company(company_id: str):
    company = Dedup.get_company(company_id)
    if not company:
        raise HTTPException(status_code=404, detail="企业不存在")
    files = Dedup.list_files(company_id=company_id)
    return JSONResponse({
        "company_id": company["id"],
        "name":       company["name"],
        "created_at": company["created_at"],
        "files": [{
            "file_id":            f["id"],
            "filename":           f["filename"],
            "file_path":          f["file_path"],
            "doc_type":           f["doc_type"],
            "doc_type_confirmed": bool(f.get("doc_type_confirmed", 0)),
            "status":             f["status"],
            "preview_text":       _build_file_preview(f["id"], f["status"]),
        } for f in files],
    })


@router.delete("/companies/{company_id}")
async def delete_company(company_id: str):
    company = Dedup.get_company(company_id)
    if not company:
        raise HTTPException(status_code=404, detail="企业不存在")
    files = Dedup.list_files(company_id=company_id)
    for f in files:
        try:
            Embedder.delete_by_file(f["id"])
        except Exception as e:
            logger.warning("[Ingest] Milvus 删除失败: %s", e)
    deleted = Dedup.delete_company(company_id)
    return JSONResponse({"status": "deleted", "company_id": company_id,
                         "name": company["name"], "files_deleted": deleted})


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

_EXT_RULES: dict[str, str] = {
    # 结构化文件：扩展名足以决定提取路线和业务类型
    ".xlsx": "table",
    ".xls": "table",
    ".csv": "table",
}


def _guess_doc_type(file_path: Path) -> str:
    suffix = file_path.suffix.lower()
    ext_type = _EXT_RULES.get(suffix)
    if ext_type:
        return ext_type

    if suffix in {".pdf", ".docx", ".docm", ".pptx"}:
        return "document"

    if suffix in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}:
        return "unknown"

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


def _build_file_preview(file_id: str, status: str, max_chars: int = 1200) -> str:
    if status != "done":
        return ""

    chunks = _query_file_chunks(file_id, limit=12)
    return _merge_chunk_text(chunks, max_chars=max_chars)


def _load_docs_for_reclassify(record: dict, limit: int = 24) -> list[Document]:
    chunks = _query_file_chunks(record["id"], limit=limit, log_prefix="Reclassify")
    docs: list[Document] = []
    ordered = sorted(chunks, key=lambda item: (item.get("chunk_index") is None, item.get("chunk_index", 0)))
    for idx, chunk in enumerate(ordered):
        text = str(chunk.get("raw_content") or chunk.get("content") or "").strip()
        if not text:
            continue
        docs.append(Document(
            page_content=text,
            metadata={
                "source": chunk.get("source") or record.get("file_path") or "",
                "label": chunk.get("label") or "text",
                "chunk_index": chunk.get("chunk_index", idx),
            },
        ))
    return docs


def _query_file_chunks(file_id: str, *, limit: int, log_prefix: str = "Companies") -> list[dict]:
    try:
        return Embedder.query_chunks_by_file(file_id=file_id, limit=limit)
    except Exception as exc:
        logger.warning("[%s] 读取 chunk 失败: file_id=%s err=%s", log_prefix, file_id[:12], exc)
        return []


def _merge_chunk_text(chunks: list[dict], max_chars: int | None) -> str:
    if not chunks:
        return ""

    ordered = sorted(chunks, key=lambda item: (item.get("chunk_index") is None, item.get("chunk_index", 0)))
    merged: list[str] = []
    total = 0

    for idx, chunk in enumerate(ordered, start=1):
        text = str(chunk.get("raw_content") or chunk.get("content") or "").strip()
        if not text:
            continue
        chunk_no = chunk.get("chunk_index")
        chunk_label = f"Chunk #{chunk_no}" if chunk_no is not None else f"Chunk {idx}"
        block = f"===== {chunk_label} =====\n{text}"
        if max_chars is not None and total >= max_chars:
            break
        if max_chars is not None:
            remaining = max_chars - total
            if len(block) > remaining:
                block = block[:remaining].rstrip() + "..."
        merged.append(block)
        total += len(block)

    return "\n\n".join(merged).strip()
    
