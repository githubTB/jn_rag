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

def process_image(image_path: str):
    docs = ExtractProcessor.extract(
        image_path,
        vl_rec_backend=settings.vl_backend,
        vl_rec_server_url=settings.vl_server_url,
        device=settings.vl_device,
        max_file_mb=settings.vl_max_file_mb,
    )
    return DocChunker.chunk(docs)
 
def get_llm_client():
    """根据 settings 初始化 LLM 客户端。"""
    from openai import OpenAI
    return OpenAI(
        base_url=settings.llm_api_base,
        api_key=settings.llm_api_key or "EMPTY",
    )
 
if __name__ == "__main__":
    # 打印当前配置（api_key 脱敏）
    print("=== 当前配置 ===")
    print(f"OCR 后端:    {settings.vl_backend or '本地CPU'}")
    print(f"OCR 地址:    {settings.vl_server_url or '无'}")
    print(f"LLM 模型:    {settings.llm_model}")
    print(f"LLM 地址:    {settings.llm_api_base}")
    print(f"Milvus:      {settings.milvus_host}:{settings.milvus_port}")
    print(f"Embedding:   {settings.embedding_model}")