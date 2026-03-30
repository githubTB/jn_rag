"""
app.py — RAG 知识库 API + 前端托管。

接口列表：
  GET    /                       前端页面
  GET    /api/health             服务健康状态（含统计）
  POST   /api/ingest             上传单文件入库
  POST   /api/ingest/scan/{id}   扫描企业目录批量入库
  GET    /api/ingest/status/{id} 查询入库状态
  GET    /api/files              已入库文件列表
  DELETE /api/files/{file_id}    删除文件
  POST   /api/companies          创建企业
  GET    /api/companies          企业列表
  GET    /api/companies/{task_id}     企业详情
  DELETE /api/companies/{task_id}     删除企业
  GET    /api/search             纯向量检索
  GET    /api/query              向量检索 + LLM 问答

启动：
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""

import warnings
warnings.filterwarnings("ignore", message="urllib3", category=Warning)
warnings.filterwarnings("ignore", message="chardet", category=Warning)

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from config.settings import settings

log_level_name = settings.log_level.upper()
log_level = getattr(logging, log_level_name, logging.INFO)

logging.basicConfig(
    level=log_level,
    format="%(asctime)s [%(levelname)5s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("服务启动:")
    logger.info("  log_level  : %s", log_level_name)
    logger.info("  upload_dir : %s", settings.upload_dir)
    logger.info("  db_path    : %s", settings.db_path)
    logger.info("  milvus     : %s:%s  collection=%s",
                settings.milvus_host, settings.milvus_port, settings.milvus_collection)
    logger.info("  embedding  : %s @ %s", settings.embedding_model, settings.embedding_device)
    logger.info("  vl_backend : %s", settings.vl_backend or "本地CPU")
    yield
    logger.info("服务关闭")


app = FastAPI(title="RAG 知识库", version="0.2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载路由
from api.routes.ingest import router as ingest_router
from api.routes.search import router as search_router
app.include_router(ingest_router)
app.include_router(search_router)

# 静态文件（JS/CSS/图片等）
if (STATIC_DIR / "assets").exists():
    app.mount("/assets", StaticFiles(directory=str(STATIC_DIR / "assets")), name="assets")


# ---------------------------------------------------------------------------
# 健康检查（前端需要的完整信息）
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health():
    from core.dedup import Dedup
    from config.settings import settings
    
    stats = Dedup.stats()

    milvus_ok = False
    vector_count = 0
    try:
        from core.embedder import Embedder
        vector_count = Embedder.count()
        milvus_ok = True
    except Exception:
        pass

    return {
        "status":        "ok" if milvus_ok else "degraded",
        "milvus":        milvus_ok,
        "milvus_host":   settings.milvus_host,
        "milvus_port":   settings.milvus_port,
        "collection":    settings.milvus_collection,
        "embed_model":   settings.embedding_model,
        "embed_device":  settings.embedding_device,
        "reranker_model":   settings.reranker_model,
        "reranker_enabled": settings.reranker_enabled,
        "llm_model":     settings.llm_model,
        "llm_api_base":  settings.llm_api_base,
        "vl_backend":    settings.vl_backend or "local_cpu",
        "vl_base_url": settings.vl_base_url or "",
        "vl_model":  settings.vl_model or "",
        "stats":         stats,
        "vector_count":  vector_count,
    }


# ---------------------------------------------------------------------------
# 前端页面
# ---------------------------------------------------------------------------

@app.get("/")
async def index():
    html = STATIC_DIR / "index.html"
    if html.exists():
        return FileResponse(str(html))
    return {"message": "请将 index.html 放到 static/ 目录"}

@app.delete("/api/collection")
async def drop_collection():
    from scripts.cleanup import _drop_milvus_collection
    from config.settings import settings
    from core.embedder import Embedder
    import core.embedder as embedder_module
    import redis

    _drop_milvus_collection(settings.milvus_collection)
    # 清 SQLite
    from core.dedup import _conn, _ensure_init
    _ensure_init()
    with _conn() as conn:
        conn.execute("DELETE FROM chunks")
        conn.execute("DELETE FROM files")
        conn.execute("DELETE FROM companies")

    # 清 Redis：broker 队列 + result backend
    redis_cleared: list[str] = []
    try:
        broker = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db_broker,
            password=settings.redis_password or None,
        )
        broker.flushdb()
        redis_cleared.append(f"broker_db={settings.redis_db_broker}")
    except Exception as exc:
        logger.warning("清理 Redis broker 失败: %s", exc)

    try:
        backend = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db_backend,
            password=settings.redis_password or None,
        )
        backend.flushdb()
        redis_cleared.append(f"backend_db={settings.redis_db_backend}")
    except Exception as exc:
        logger.warning("清理 Redis backend 失败: %s", exc)

    # 清进程内缓存，确保当前 API 进程也拿到新 schema
    embedder_module._collection = None

    # 立即触发重建，确保集合结构和名称恢复到当前配置值。
    Embedder.count()
    return {
        "message": "知识库、SQLite 和 Redis 队列已清空并重建",
        "collection": settings.milvus_collection,
        "redis_cleared": redis_cleared,
    }
    
