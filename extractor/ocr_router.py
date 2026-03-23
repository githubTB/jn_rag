"""
extractor/ocr_router.py — 图片 OCR 路由。

所有图片统一走远程 PaddleOCR-VL（vLLM 服务），本地不加载任何模型。

.env 配置：
    VL_BACKEND=vllm-server
    VL_SERVER_URL=http://<服务器IP>:8118/v1
"""

from __future__ import annotations

import logging
from pathlib import Path

from models.document import Document

logger = logging.getLogger(__name__)


def route_ocr(
    file_path: str,
    doc_type: str = "unknown",
    **kwargs,
) -> list[Document]:
    """所有图片走远程 PaddleOCR-VL。"""
    logger.info("[OCRRouter] %s  doc_type=%s", Path(file_path).name, doc_type)

    from extractor.image_extractor import ImageExtractor
    # doc_type 传 unknown 防止 ImageExtractor 内部再次触发路由（避免循环）
    extractor = ImageExtractor(file_path, doc_type="unknown", **kwargs)
    return extractor.extract()
    