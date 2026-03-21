"""
api/routes/ingest.py — 文件上传入库 + 企业管理接口。

接口
----
POST   /api/ingest                         上传单个文件入库
POST   /api/ingest/scan/{company_id}       扫描企业目录批量入库
GET    /api/ingest/status/{file_id}        查询入库状态
GET    /api/files                          已入库文件列表
DELETE /api/files/{file_id}                删除文件

POST   /api/companies                      创建企业
GET    /api/companies                      企业列表
GET    /api/companies/{company_id}         企业详情（含文件列表）
DELETE /api/companies/{company_id}         删除企业及其所有文件

文件存储结构（两种方式都支持）
------------------------------
方式一：按企业建目录（批量导入）
    uploaded_files/
    ├── one/        ← company_id=one
    │   ├── 营业执照.jpg
    │   └── 合同.pdf
    └── two/        ← company_id=two
        └── 发票.jpg

    # 扫描入库
    POST /api/ingest/scan/one?doc_type=license

方式二：API 上传（带参数）
    POST /api/ingest?company_id=one&doc_type=license
    -F "file=@营业执照.jpg"
    # 文件自动保存到 uploaded_files/one/营业执照.jpg
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
    """
    上传文件并入库。

    Parameters
    ----------
    company_id : 所属企业 ID（需先通过 POST /api/companies 创建）
    doc_type   : 文件类型（不传则根据文件名自动推断）。
                 可选值：license / invoice / table / nameplate / document
    sync       : True = 同步等待结果；False（默认）= 后台异步处理
    """
    # 校验企业存在
    if company_id and not Dedup.get_company(company_id):
        raise HTTPException(
            status_code=404,
            detail=f"企业不存在: {company_id}，请先通过 POST /api/companies 创建",
        )

    # doc_type 未传时用文件名自动推断
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
        raise HTTPException(
            status_code=415,
            detail=f"不支持的文件格式: {ext}",
        )

    # 保存文件：有 company_id 时存到子目录 uploaded_files/{company_id}/
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
    logger.info("[Ingest] 保存: %s (%.2f MB)  company=%s  type=%s",
                save_path.name, size_mb, company_id, doc_type)

    # 文件级去重
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

    # 触发流水线
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
# POST /api/ingest/scan/{company_id} — 扫描目录批量入库
# ---------------------------------------------------------------------------

# 支持的扩展名（扫描时识别）
_SCAN_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif",
    ".pdf", ".docx", ".pptx", ".xlsx", ".xls", ".csv",
    ".txt", ".md", ".html", ".htm",
    # .doc / .docm 由预处理转换为 .docx 后处理（见 _prepare_doc）
}

# 跳过的文件/目录
_SKIP_FILES = {".DS_Store", "Thumbs.db", ".gitkeep"}
_SKIP_SUFFIXES = {".dwg", ".dxf", ".ds_store"}  # CAD 文件等不支持的格式

# 路径关键词推断规则（优先级从高到低，路径含目录名一起判断）
_PATH_RULES: list[tuple[list[str], str]] = [
    # license
    (["营业执照", "执照", "license", "证照", "信用中国"],                        "license"),
    (["排污许可", "许可证"],                                                      "license"),
    # nameplate（设备铭牌、现场照片优先于其他规则）
    (["现场照片", "铭牌", "nameplate", "设备铭牌"],                               "nameplate"),
    (["微信图片", "现场"],                                                         "nameplate"),
    # table
    (["统计表", "汇总", "台账", "明细", "一览表", "固废处理统计", "废水废气"],    "table"),
    (["水电气", "生产设备"],                                                       "table"),
    # document（报告、合同、制度、证书等）
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
    """
    根据文件完整路径（含父目录名）推断 doc_type。
    用完整路径而不只是文件名，目录名往往含有更多语义。
    """
    # 把路径各部分拼成一个字符串一起匹配
    path_str = "/".join(file_path.parts).lower()

    for keywords, dt in _PATH_RULES:
        if any(k.lower() in path_str for k in keywords):
            return dt

    # 兜底：图片默认 nameplate（现场照片可能性大），文档默认 document
    suffix = file_path.suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}:
        return "nameplate"
    return "document"


def _prepare_doc(file_path: Path) -> Path | None:
    """
    处理不直接支持的格式：
    - .doc / .docm → 尝试用 python-docx 读取（部分 .doc 兼容）
    - .dwg → 跳过
    返回实际要处理的路径，None 表示跳过。
    """
    suffix = file_path.suffix.lower()
    if suffix in _SKIP_SUFFIXES:
        return None
    if suffix == ".docm":
        # .docm 是带宏的 docx，直接改后缀尝试读取
        return file_path  # WordExtractor 能处理
    if suffix == ".doc":
        # 优先用同名 .docx（GMP 目录里 .doc 和 .docx 并存）
        docx_path = file_path.with_suffix(".docx")
        if docx_path.exists():
            logger.debug("[Scan] .doc 有对应 .docx，跳过: %s", file_path.name)
            return None  # 跳过 .doc，只处理 .docx
        # 无对应 .docx 时，直接处理 .doc（WordExtractor 部分兼容）
        return file_path
    return file_path


@router.post("/ingest/scan/{company_id}")
async def scan_company_dir(
    company_id: str,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    doc_type: str | None = None,
    sync: bool = False,
):
    """
    扫描企业目录下的所有文件并批量入库。

    扫描路径：uploaded_files/{company_id}/

    Parameters
    ----------
    doc_type : 强制指定所有文件的类型（可选）。
               不传时根据【目录名 + 文件名】自动推断，推断规则：
               - 路径含"营业执照/执照/信用中国"     → license
               - 路径含"设备/铭牌/现场照片/微信图片" → nameplate
               - 路径含"统计/汇总/台账/明细/水电气"  → table
               - 路径含"报告/合同/制度/证书/环评"    → document
               - 发票/单据                           → invoice
               - 无法推断的图片                      → nameplate（兜底）
               - 无法推断的文档                      → document（兜底）
    sync     : True = 同步等待所有文件处理完（文件多时会很慢，谨慎使用）

    示例
    ----
        # 扫描 one 目录，文件类型自动推断
        POST /api/ingest/scan/one

        # 强制所有文件都用 license 类型
        POST /api/ingest/scan/one?doc_type=license
    """
    # 校验企业
    if not Dedup.get_company(company_id):
        raise HTTPException(
            status_code=404,
            detail=f"企业不存在: {company_id}，请先通过 POST /api/companies 创建",
        )

    scan_dir = Path(settings.upload_dir) / company_id
    if not scan_dir.exists():
        raise HTTPException(
            status_code=404,
            detail=f"目录不存在: {scan_dir}，请先将文件放入该目录",
        )

    # 递归收集所有支持的文件（过滤 .DS_Store、.dwg 等）
    all_files: list[Path] = []
    for f in scan_dir.rglob("*"):
        if not f.is_file():
            continue
        if f.name in _SKIP_FILES:
            continue
        suffix = f.suffix.lower()
        if suffix in _SKIP_SUFFIXES:
            logger.debug("[Scan] 跳过不支持格式: %s", f.name)
            continue
        # .doc 有对应 .docx 则跳过
        prepared = _prepare_doc(f)
        if prepared is None:
            continue
        # 只收集支持的格式（含 .doc .docm）
        supported = _SCAN_EXTENSIONS | {".doc", ".docm"}
        if suffix not in supported:
            logger.debug("[Scan] 跳过未知格式: %s", f.name)
            continue
        all_files.append(prepared)

    if not all_files:
        return JSONResponse({
            "status":     "empty",
            "company_id": company_id,
            "scan_dir":   str(scan_dir),
            "message":    "目录下没有找到支持的文件",
        })

    # 为每个文件确定 doc_type，同时按是否需要 OCR 分两个队列
    # 文档类（docx/xlsx/pdf/csv/txt）：直接读文字，快，优先处理
    # 图片类（jpg/png/tiff 等）：需要远程 vLLM，慢，串行排队
    _DOC_SUFFIXES = {".docx", ".doc", ".docm", ".xlsx", ".xls", ".csv",
                     ".txt", ".md", ".html", ".htm", ".pptx", ".pdf"}
    _IMG_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}

    doc_tasks: list[dict] = []   # 文档类：优先同步处理
    img_tasks: list[dict] = []   # 图片类：串行异步排队

    for f in all_files:
        dt = doc_type or _guess_doc_type(f)
        task = {"path": str(f), "doc_type": dt}
        logger.info("[Scan] %s  →  %s", f.relative_to(scan_dir), dt)
        if f.suffix.lower() in _DOC_SUFFIXES:
            doc_tasks.append(task)
        else:
            img_tasks.append(task)

    all_tasks = doc_tasks + img_tasks
    logger.info("[Scan] company=%s 文档类=%d 图片类=%d",
                company_id, len(doc_tasks), len(img_tasks))

    if sync:
        # 同步：文档类 → 图片类，逐个处理
        results = []
        for task in all_tasks:
            result = IngestTask.run(
                task["path"],
                company_id=company_id,
                doc_type=task["doc_type"],
            )
            results.append({
                "file":     Path(task["path"]).name,
                "doc_type": task["doc_type"],
                "status":   result.get("status"),
                "chunks":   result.get("new_chunks", 0),
                "error":    result.get("error"),
            })
        return JSONResponse({
            "status":     "done",
            "company_id": company_id,
            "total":      len(all_tasks),
            "doc_count":  len(doc_tasks),
            "img_count":  len(img_tasks),
            "results":    results,
        })
    else:
        # 异步：文档类立即在 background 串行处理；图片类排在后面串行
        # 同一个 background task 保证顺序：文档先，图片后，不打爆 vLLM
        def _run_serial(task_list: list[dict], cid: str) -> None:
            for t in task_list:
                try:
                    IngestTask.run(t["path"], company_id=cid, doc_type=t["doc_type"])
                except Exception as exc:
                    logger.error("[Scan] 文件处理异常: %s  %s", t["path"], exc)

        background_tasks.add_task(_run_serial, all_tasks, company_id)
        return JSONResponse({
            "status":     "processing",
            "company_id": company_id,
            "total":      len(all_tasks),
            "doc_count":  len(doc_tasks),
            "img_count":  len(img_tasks),
            "files":      [
                {"file": Path(t["path"]).name, "doc_type": t["doc_type"],
                 "queue": "doc" if t in doc_tasks else "img"}
                for t in all_tasks
            ],
            "message":    (
                f"共 {len(all_tasks)} 个文件：文档类 {len(doc_tasks)} 个优先处理，"
                f"图片类 {len(img_tasks)} 个排队（vLLM 串行，每次 1 张）"
            ),
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
        "file_id":    record["id"],
        "filename":   record["filename"],
        "company_id": record["company_id"],
        "doc_type":   record["doc_type"],
        "status":     record["status"],
        "created_at": record["created_at"],
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
                "file_id":    f["id"][:16] + "..",
                "filename":   f["filename"],
                "company_id": f["company_id"],
                "doc_type":   f["doc_type"],
                "status":     f["status"],
                "created_at": f["created_at"],
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
    return JSONResponse({
        "status":   "deleted",
        "file_id":  file_id,
        "filename": record["filename"],
    })


# ---------------------------------------------------------------------------
# 企业管理
# ---------------------------------------------------------------------------

class CompanyCreate(BaseModel):
    company_id: str
    name: str


@router.post("/companies", status_code=201)
async def create_company(body: CompanyCreate):
    """创建或更新企业信息。"""
    Dedup.register_company(body.company_id, body.name)
    return JSONResponse({
        "status":     "ok",
        "company_id": body.company_id,
        "name":       body.name,
    }, status_code=201)


@router.get("/companies")
async def list_companies():
    companies = Dedup.list_companies()
    result = []
    for c in companies:
        files = Dedup.list_files(company_id=c["id"])
        result.append({
            "company_id": c["id"],
            "name":       c["name"],
            "created_at": c["created_at"],
            "file_count": len(files),
            "done_count": sum(1 for f in files if f["status"] == "done"),
        })
    return JSONResponse({"total": len(result), "companies": result})


@router.get("/companies/{company_id}")
async def get_company(company_id: str):
    """企业详情，含所有文件列表。"""
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
                "file_id":  f["id"][:16] + "..",
                "filename": f["filename"],
                "doc_type": f["doc_type"],
                "status":   f["status"],
            }
            for f in files
        ],
    })


@router.delete("/companies/{company_id}")
async def delete_company(company_id: str):
    """删除企业及其所有文件的向量和记录。"""
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
        "status":       "deleted",
        "company_id":   company_id,
        "name":         company["name"],
        "files_deleted": deleted,
    })
    