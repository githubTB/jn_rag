"""
app.py — RAG 知识库 API 入口。

接口列表：
  GET    /api/health              服务健康状态
  POST   /api/ingest              上传文件入库（支持 sync 参数）
  GET    /api/ingest/status/{id}  查询入库状态
  GET    /api/files               已入库文件列表
  DELETE /api/files/{file_id}     删除文件

启动：
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""

import warnings
warnings.filterwarnings("ignore", message="urllib3", category=Warning)
warnings.filterwarnings("ignore", message="chardet", category=Warning)

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config.settings import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)5s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 启动 / 关闭生命周期
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时预热（可选，提前加载模型避免第一次请求慢）
    logger.info("服务启动，配置:")
    logger.info("  upload_dir : %s", settings.upload_dir)
    logger.info("  db_path    : %s", settings.db_path)
    logger.info("  milvus     : %s:%s", settings.milvus_host, settings.milvus_port)
    logger.info("  embedding  : %s @ %s", settings.embedding_model, settings.embedding_device)
    logger.info("  vl_backend : %s", settings.vl_backend or "本地CPU")
    logger.info("  vl_url     : %s", settings.vl_server_url or "无")
    yield
    logger.info("服务关闭")


# ---------------------------------------------------------------------------
# FastAPI 应用
# ---------------------------------------------------------------------------

app = FastAPI(
    title="RAG 知识库 API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载路由
from api.routes.ingest import router as ingest_router
app.include_router(ingest_router)


# ---------------------------------------------------------------------------
# 健康检查
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health():
    from core.dedup import Dedup
    stats = Dedup.stats()
    return {
        "status": "ok",
        "stats":  stats,
    }