"""
settings.py — 项目全局配置。

优先级：环境变量 > .env 文件 > 默认值

.env 示例
---------
    # 存储
    UPLOAD_DIR=uploaded_files
    DB_PATH=rag_meta.db

    # Milvus
    MILVUS_HOST=localhost
    MILVUS_PORT=19530
    MILVUS_COLLECTION=rag_docs

    # Embedding
    EMBEDDING_MODEL=BAAI/bge-m3
    EMBEDDING_DEVICE=cpu

    # LLM
    LLM_PROVIDER=openai
    LLM_API_BASE=http://117.x.x.x:8002/v1
    LLM_API_KEY=your_key
    LLM_MODEL=Qwen3.5-27B
    LLM_TEMPERATURE=0.3
    LLM_MAX_TOKENS=2048

    # OCR
    VL_BACKEND=vllm-server
    VL_SERVER_URL=http://117.x.x.x:8118/v1
    VL_DEVICE=
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,   # 环境变量大小写不敏感
        extra="ignore",         # 忽略 .env 里多余的 key，不报错
    )

    # ── 文件存储 ──────────────────────────────────────────────────────
    upload_dir: str = Field("uploaded_files", description="上传文件目录")
    db_path: str = Field("rag_meta.db", description="SQLite 元数据库路径")

    # ── Milvus 向量库 ─────────────────────────────────────────────────
    milvus_host: str = Field("localhost", description="Milvus 地址")
    milvus_port: int = Field(19530, description="Milvus 端口")
    milvus_collection: str = Field("rag_docs", description="集合名称")

    # ── Embedding ─────────────────────────────────────────────────────
    embedding_model: str = Field("BAAI/bge-m3", description="向量模型名称")
    embedding_device: str = Field("cpu", description="向量模型推理设备")
    embedding_batch_size: int = Field(32, description="批量向量化大小")

    # ── LLM ──────────────────────────────────────────────────────────
    llm_provider: str = Field("openai", description="openai / ollama")
    llm_api_base: str = Field("", description="LLM API 地址")
    llm_api_key: str = Field("", description="LLM API Key")
    llm_model: str = Field("Qwen3.5-27B", description="模型名称")
    llm_temperature: float = Field(0.3, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(2048, ge=1)

    # ── OCR（PaddleOCR-VL）───────────────────────────────────────────
    vl_backend: str | None = Field(None, description="none=本地cpu / vllm-server 等")
    vl_server_url: str | None = Field(None, description="vLLM 服务地址")
    vl_device: str | None = Field(None, description="OCR 推理设备，None=自动")
    vl_max_file_mb: float = Field(50.0, description="OCR 图片大小上限 MB")


# 全局单例，项目任意位置 from config.settings import settings 即可使用
settings = Settings()
