"""
extractor/ocr_router.py — 图片 OCR 引擎路由。

策略：所有图片统一走 PaddleOCR-VL（远程 vLLM 服务）
------------------------------------------------------
- 不在本地加载任何模型（避免 Mac 内存爆炸）
- 营业执照 / 发票 / 表格 / 铭牌 全部走 VL-1.5-0.9B
- nameplate 已部署 GOT-OCR2 时走 GOT-OCR2，否则也走 VL
- 并发控制：通过 vl_rec_max_concurrency 限制同时请求数

.env 配置：
    VL_BACKEND=vllm-server
    VL_SERVER_URL=http://<Ubuntu-IP>:8118/v1
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
    """
    所有图片路由到合适的 OCR 引擎。

    nameplate + GOT-OCR2 已部署 → GOT-OCR2
    其他所有类型               → PaddleOCR-VL（远程 vLLM）
    """
    engine = "got_ocr" if (doc_type == "nameplate" and _got_ocr_available()) else "vl"

    logger.info("[OCRRouter] %s  doc_type=%s  engine=%s",
                Path(file_path).name, doc_type, engine)

    if engine == "got_ocr":
        return _run_got_ocr(file_path, **kwargs)
    else:
        return _run_paddleocr_vl(file_path, doc_type=doc_type, **kwargs)


def _got_ocr_available() -> bool:
    try:
        from config.settings import settings
        return settings.got_ocr_available
    except Exception:
        return False


# ---------------------------------------------------------------------------
# PaddleOCR-VL（远程 vLLM，所有图片的主引擎）
# ---------------------------------------------------------------------------

def _run_paddleocr_vl(file_path: str, doc_type: str = "unknown", **kwargs) -> list[Document]:
    """
    调用远程 vLLM 上的 PaddleOCR-VL-1.5-0.9B。

    并发控制：vl_rec_max_concurrency 默认为 1，
    批量扫描时不要同时发多个请求，让 vLLM 一个一个处理，
    避免显存 OOM 或响应超时。
    """
    from extractor.image_extractor import ImageExtractor
    # doc_type 传 unknown 防止 ImageExtractor 内部再次触发路由（避免循环）
    extractor = ImageExtractor(file_path, doc_type="unknown", **kwargs)
    return extractor.extract()


# ---------------------------------------------------------------------------
# GOT-OCR2（可选，仅 nameplate 且已部署时使用）
# ---------------------------------------------------------------------------

def _run_got_ocr(file_path: str, **_ignored) -> list[Document]:
    from config.settings import settings
    model_path = settings.got_ocr_model or "ucaslcl/GOT-OCR2_0"
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            device_map="cuda",
            use_safetensors=True,
            pad_token_id=tokenizer.eos_token_id,
            trust_remote_code=True,
        ).eval()

        with torch.no_grad():
            res = model.chat(tokenizer, file_path, ocr_type="ocr")

        logger.info("[OCRRouter] GOT-OCR2 完成: %d 字符", len(res))
        return [Document(
            page_content=res.strip(),
            metadata={"source": file_path, "label": "ocr", "engine": "got_ocr2"},
        )]
    except Exception as exc:
        logger.error("[OCRRouter] GOT-OCR2 失败，降级 VL: %s", exc)
        return _run_paddleocr_vl(file_path)
        