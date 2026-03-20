"""
app.py — RAG 知识库 API

接口：
  GET  /api/health          服务健康状态
  POST /api/ingest          上传文件 → 解析 → 向量化 → 入库（自动去重）
  GET  /api/search          纯向量检索（不走 LLM）
  GET  /api/query           向量检索 + LLM（qwen3.5） 生成带来源标注的答案
  GET  /api/files           已入库文件列表
  DELETE /api/files/{name}  删除文件
"""

import warnings

# 过滤 requests/urllib3 版本不匹配的无害警告
warnings.filterwarnings("ignore", message="urllib3", category=Warning)
warnings.filterwarnings("ignore", message="chardet", category=Warning)

from config.settings import settings
from extract_processor import ExtractProcessor
from extractor.doc_chunker import DocChunker