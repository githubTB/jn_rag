"""
api/routes/ingest.py — 文件上传入库 + 企业管理接口。

接口
----
POST   /api/ingest                         上传单个文件入库
POST   /api/ingest/scan/{company_id}       扫描企业目录批量入库
GET    /api/ingest/status/{file_id}        查询入库状态
GET    /api/files                          已入库文件列表
DELETE /api/files/{file_id}                删除文件
PATCH  /api/files/{file_id}/doc_type       人工修改文件类型（带确认标记）
POST   /api/files/{file_id}/reprocess      重新处理文件（删旧向量→重新解析入库）

POST   /api/companies                      创建企业
GET    /api/companies                      企业列表
GET    /api/companies/{company_id}         企业详情（含文件列表）
DELETE /api/companies/{company_id}         删除企业及其所有文件
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
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
# POST /api/ingest
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
        raise HTTPException(
            status_code=404,
            detail=f"企业不存在: {company_id}，请先通过 POST /api/companies 创建",
        )

    if doc_type is None:
        doc_type = _guess_doc_type(Path(file.filename or "unknown"))
        logger.info("[Ingest] 自动推断 doc_type=%s  文件=%s", doc_type, file.filename)
    elif doc_type not in DocType.ALL:
        raise HTTPException(
            status_code=400,
            detail=f"无效的 doc_type: {doc_type}，可选: {', '.join(sorted(DocType.ALL))}",
        )

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
        return JSONResponse({
            "status":     "skipped",
            "reason":     "文件内容已入库",
            "file_id":    file_id,
            "file":       existing.get("filename", filename),
            "company_id": existing.get("company_id"),
            "doc_type":   existing.get("doc_type"),
        })

    kwargs = dict(company_id=company_id, doc_type=doc_type)
    if sync:
        result = IngestTask.run(str(save_path), **kwargs)
        return JSONResponse(result)
    else:
        background_tasks.add_task(IngestTask.run, str(save_path), **kwargs)
        return JSONResponse({
            "status":     "processing",
            "file_id":    file_id,
            "file":       save_path.name,
            "company_id": company_id,
            "doc_type":   doc_type,
            "message":    "文件已上传，正在后台处理",
        }, status_code=202)


# ---------------------------------------------------------------------------
# POST /api/ingest/scan/{company_id}
# ---------------------------------------------------------------------------

_SCAN_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif",
    ".pdf", ".docx", ".pptx", ".xlsx", ".xls", ".csv",
    ".txt", ".md", ".html", ".htm",
}
_SKIP_FILES    = {".DS_Store", "Thumbs.db", ".gitkeep"}
_SKIP_SUFFIXES = {".dwg", ".dxf", ".ds_store"}

_PATH_RULES: list[tuple[list[str], str]] = [
    (["营业执照", "执照", "license", "证照", "信用中国"],                        "license"),
    (["排污许可", "许可证"],                                                      "license"),
    (["现场照片", "铭牌", "nameplate", "设备铭牌"],                               "nameplate"),
    (["微信图片", "现场"],                                                         "nameplate"),
    (["统计表", "汇总", "台账", "明细", "一览表", "固废处理统计", "废水废气"],    "table"),
    (["水电气", "生产设备"],                                                       "table"),
    (["检测报告", "监测报告", "废气", "废水", "噪音", "环境影响", "环评",
      "排污", "验收", "竣工", "锅炉", "分析报告"],                               "document"),
    (["认证", "证书", "资质", "许可", "批准书", "许可证"],                        "document"),
    (["合同", "协议", "制度", "规程", "工艺"],                                     "document"),
    (["总平面图", "平面图", "报告书", "报告"],                                      "document"),
    (["收资清单", "评估报告", "诊断报告", "GMP", "AL-SMP"],                       "document"),
    (["危废", "固废", "废品", "污水"],                                             "document"),
    (["发票", "invoice", "单据", "收据"],                                          "invoice"),
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
    if suffix in _SKIP_SUFFIXES:
        return None
    if suffix == ".doc":
        docx_path = file_path.with_suffix(".docx")
        if docx_path.exists():
            return None
        return file_path
    return file_path


@router.post("/ingest/scan/{company_id}")
async def scan_company_dir(
    company_id: str,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    doc_type: str | None = None,
    sync: bool = False,
):
    if not Dedup.get_company(company_id):
        raise HTTPException(status_code=404, detail=f"企业不存在: {company_id}")

    scan_dir = Path(settings.upload_dir) / company_id
    if not scan_dir.exists():
        raise HTTPException(status_code=404, detail=f"目录不存在: {scan_dir}")

    all_files: list[Path] = []
    for f in scan_dir.rglob("*"):
        if not f.is_file():
            continue
        if f.name in _SKIP_FILES:
            continue
        if f.suffix.lower() in _SKIP_SUFFIXES:
            continue
        prepared = _prepare_doc(f)
        if prepared is None:
            continue
        supported = _SCAN_EXTENSIONS | {".doc", ".docm"}
        if f.suffix.lower() not in supported:
            continue
        all_files.append(prepared)

    if not all_files:
        return JSONResponse({
            "status": "empty", "company_id": company_id,
            "scan_dir": str(scan_dir), "message": "目录下没有找到支持的文件",
        })

    tasks: list[dict] = []
    for f in all_files:
        dt = doc_type or _guess_doc_type(f)
        tasks.append({"path": str(f), "doc_type": dt})

    if sync:
        results = []
        for task in tasks:
            result = IngestTask.run(task["path"], company_id=company_id, doc_type=task["doc_type"])
            results.append({
                "file":     Path(task["path"]).name,
                "doc_type": task["doc_type"],
                "status":   result.get("status"),
                "chunks":   result.get("new_chunks", 0),
            })
        return JSONResponse({"status": "done", "company_id": company_id,
                             "total": len(tasks), "results": results})
    else:
        for task in tasks:
            background_tasks.add_task(IngestTask.run, task["path"],
                                      company_id=company_id, doc_type=task["doc_type"])
        return JSONResponse({
            "status": "processing", "company_id": company_id, "total": len(tasks),
            "files": [{"file": Path(t["path"]).name, "doc_type": t["doc_type"]} for t in tasks],
            "message": f"已提交 {len(tasks)} 个文件到后台处理",
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
        "file_id":             record["id"],
        "filename":            record["filename"],
        "company_id":          record["company_id"],
        "doc_type":            record["doc_type"],
        "doc_type_confirmed":  bool(record.get("doc_type_confirmed", 0)),
        "status":              record["status"],
        "created_at":          record["created_at"],
    }
    if record["status"] == "done":
        try:
            resp["total_vectors"] = Embedder.count()
        except Exception:
            pass
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
        "files": [
            {
                "file_id":            f["id"][:16] + "..",
                "filename":           f["filename"],
                "company_id":         f["company_id"],
                "doc_type":           f["doc_type"],
                "doc_type_confirmed": bool(f.get("doc_type_confirmed", 0)),
                "status":             f["status"],
                "created_at":         f["created_at"],
            }
            for f in files
        ],
    })


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
    return JSONResponse({"status": "deleted", "file_id": file_id, "filename": record["filename"]})


# ---------------------------------------------------------------------------
# PATCH /api/files/{file_id}/doc_type — 人工修改文件类型
# ---------------------------------------------------------------------------

class DocTypeUpdate(BaseModel):
    doc_type: str
    confirmed: bool = True


@router.patch("/files/{file_id}/doc_type")
async def update_doc_type(file_id: str, body: DocTypeUpdate):
    """
    人工修改文件的 doc_type，并标记为已确认。

    注意：修改 doc_type 后需要手动触发 reprocess 才能用新类型重新解析。
    """
    if body.doc_type not in DocType.ALL:
        raise HTTPException(
            status_code=400,
            detail=f"无效的 doc_type: {body.doc_type}，可选: {', '.join(sorted(DocType.ALL))}",
        )

    # 支持前缀匹配（file_id 可能是截断的）
    record = Dedup.get_file(file_id)
    if not record:
        all_files = Dedup.list_files()
        matched = [f for f in all_files if f["id"].startswith(file_id)]
        if len(matched) == 1:
            record = matched[0]
            file_id = record["id"]
        elif len(matched) > 1:
            raise HTTPException(status_code=400, detail="file_id 前缀匹配到多个文件，请提供完整 id")
        else:
            raise HTTPException(status_code=404, detail="文件不存在")

    old_type = record["doc_type"]
    success = Dedup.update_doc_type(file_id, body.doc_type, confirmed=body.confirmed)
    if not success:
        raise HTTPException(status_code=404, detail="文件不存在")

    logger.info("[Ingest] 人工修改 doc_type: %s → %s  file=%s",
                old_type, body.doc_type, record["filename"])

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
# POST /api/files/{file_id}/reprocess — 重新处理文件
# ---------------------------------------------------------------------------

@router.post("/files/{file_id}/reprocess")
async def reprocess_file(
    file_id: str,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    sync: bool = False,
):
    """
    重新处理文件：删除旧向量和 chunk 记录，用当前 doc_type 重新解析入库。

    流程（方案A）：
        1. 删除 Milvus 中该文件的全部向量
        2. 删除 SQLite 中该文件的全部 chunk 记录
        3. 重置文件状态为 pending
        4. 用当前 doc_type（含人工确认值）重新触发完整入库流水线

    适用场景：
        - 人工修改 doc_type 后希望用新类型重新解析
        - 文件解析失败后重试
        - OCR 结果不满意，切换后端后重新处理
    """
    # 支持前缀匹配
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
        raise HTTPException(
            status_code=404,
            detail=f"原始文件不存在: {file_path}，无法重新处理",
        )

    logger.info("[Reprocess] 开始重新处理: %s  doc_type=%s  confirmed=%s",
                record["filename"], record["doc_type"],
                bool(record.get("doc_type_confirmed", 0)))

    # ── STEP 1: 删除旧向量（Milvus）──────────────────────────────────
    try:
        Embedder.delete_by_file(file_id)
        logger.info("[Reprocess] 旧向量已删除: file_id=%s", file_id[:12])
    except Exception as e:
        logger.warning("[Reprocess] Milvus 删除失败（继续处理）: %s", e)

    # ── STEP 2: 删除旧 chunk 记录（SQLite）───────────────────────────
    deleted_chunks = Dedup.delete_chunks_by_file(file_id)
    logger.info("[Reprocess] 旧 chunk 记录已删除: %d 条", deleted_chunks)

    # ── STEP 3: 重置文件状态，保留 doc_type 和 confirmed 标记 ────────
    with Dedup._get_conn_ctx() as conn:
        conn.execute(
            "UPDATE files SET status = 'pending' WHERE id = ?", (file_id,)
        )

    # ── STEP 4: 触发重新入库（force=True 跳过文件级去重）────────────
    kwargs = dict(
        company_id=record["company_id"],
        doc_type=record["doc_type"],
        force=True,
    )

    if sync:
        result = IngestTask.run(str(file_path), **kwargs)
        return JSONResponse({
            **result,
            "reprocessed":        True,
            "deleted_chunks":     deleted_chunks,
            "doc_type_confirmed": bool(record.get("doc_type_confirmed", 0)),
        })
    else:
        background_tasks.add_task(IngestTask.run, str(file_path), **kwargs)
        return JSONResponse({
            "status":             "processing",
            "file_id":            file_id,
            "filename":           record["filename"],
            "doc_type":           record["doc_type"],
            "doc_type_confirmed": bool(record.get("doc_type_confirmed", 0)),
            "deleted_chunks":     deleted_chunks,
            "reprocessed":        True,
            "message":            "已清除旧数据，正在后台重新处理",
        }, status_code=202)


# ---------------------------------------------------------------------------
# 企业管理
# ---------------------------------------------------------------------------

class CompanyCreate(BaseModel):
    company_id: str
    name: str


@router.post("/companies", status_code=201)
async def create_company(body: CompanyCreate):
    Dedup.register_company(body.company_id, body.name)
    return JSONResponse({
        "status": "ok", "company_id": body.company_id, "name": body.name,
    }, status_code=201)


@router.get("/companies")
async def list_companies():
    companies = Dedup.list_companies()
    result = []
    for c in companies:
        files = Dedup.list_files(company_id=c["id"])
        result.append({
            "company_id":       c["id"],
            "name":             c["name"],
            "created_at":       c["created_at"],
            "file_count":       len(files),
            "done_count":       sum(1 for f in files if f["status"] == "done"),
            "confirmed_count":  sum(1 for f in files if f.get("doc_type_confirmed")),
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
        "files": [
            {
                "file_id":            f["id"],
                "filename":           f["filename"],
                "doc_type":           f["doc_type"],
                "doc_type_confirmed": bool(f.get("doc_type_confirmed", 0)),
                "status":             f["status"],
            }
            for f in files
        ],
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
    return JSONResponse({
        "status": "deleted", "company_id": company_id,
        "name": company["name"], "files_deleted": deleted,
    })