"""
extractor/ocr_router.py — 图片 OCR 引擎路由。

根据 doc_type 选择最合适的 OCR 引擎，不改变上层调用方式。

路由策略
--------
license / invoice / document  → PP-OCRv4（CPU，轻量，印刷文字准确率高）
table                          → PP-OCRv4（后续可加 TableRec 结构化）
nameplate / unknown            → PaddleOCR-VL（现阶段兜底，后续替换 GOT-OCR2）

为什么现在还用 PaddleOCR-VL 兜底：
- GOT-OCR2 还未部署，等部署好后把 _run_got_ocr 填完即可
- nameplate（设备铭牌、现场照片）光线复杂，PP-OCRv4 准确率不够
- unknown 类型不确定，用能力更强的模型兜底

替换 GOT-OCR2 时只需修改 _run_got_ocr，上层无感知。

使用方式（外部不需要直接调用，由 ImageExtractor 调用）：
    from extractor.ocr_router import route_ocr
    docs = route_ocr(file_path, doc_type="invoice")
"""

from __future__ import annotations

import logging
from pathlib import Path

from models.document import Document

logger = logging.getLogger(__name__)

# doc_type → OCR 引擎映射
_PPOCR_TYPES  = {"license", "invoice", "document", "table"}
_GOTOCR_TYPES = {"nameplate"}
_FALLBACK     = "paddleocr_vl"  # unknown 的兜底


def route_ocr(
    file_path: str,
    doc_type: str = "unknown",
    **kwargs,
) -> list[Document]:
    """
    根据 doc_type 路由到对应 OCR 引擎，返回 Document 列表。

    Parameters
    ----------
    file_path : 图片文件路径
    doc_type  : 文件类型（license/invoice/table/nameplate/document/unknown）
    **kwargs  : 透传给底层 OCR 引擎的参数
    """
    engine = _select_engine(doc_type)
    logger.info("[OCRRouter] file=%s  doc_type=%s  engine=%s",
                Path(file_path).name, doc_type, engine)

    if engine == "ppocr":
        return _run_ppocr(file_path, doc_type=doc_type, **kwargs)
    elif engine == "got_ocr":
        return _run_got_ocr(file_path, **kwargs)
    else:
        return _run_paddleocr_vl(file_path, **kwargs)


def _select_engine(doc_type: str) -> str:
    if doc_type in _PPOCR_TYPES:
        return "ppocr"
    if doc_type in _GOTOCR_TYPES:
        # GOT-OCR2 未部署时降级到 paddleocr_vl
        if _got_ocr_available():
            return "got_ocr"
        logger.warning("[OCRRouter] GOT-OCR2 未部署，降级到 PaddleOCR-VL")
        return "paddleocr_vl"
    return "paddleocr_vl"  # unknown 兜底


def _got_ocr_available() -> bool:
    """检查 GOT-OCR2 是否已部署可用（从 settings.got_ocr_model 读路径）。"""
    try:
        from config.settings import settings
        return settings.got_ocr_available
    except Exception:
        return False


# ---------------------------------------------------------------------------
# PP-OCRv4 引擎
# ---------------------------------------------------------------------------

_ppocr_instance = None


def _get_ppocr():
    global _ppocr_instance
    if _ppocr_instance is None:
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise ImportError("paddleocr 未安装: pip install paddleocr") from exc

        logger.info("[OCRRouter] 初始化 PP-OCRv4 (CPU)")
        try:
            _ppocr_instance = PaddleOCR(
                use_textline_orientation=True,
                lang="ch",
                ocr_version="PP-OCRv4",
                cpu_threads=4,
                enable_mkldnn=False,
                det_db_thresh=0.3,
                det_db_box_thresh=0.4,
                det_db_unclip_ratio=1.8,
                det_limit_side_len=4000,
            )
        except TypeError:
            # 旧版 paddleocr 参数不同
            _ppocr_instance = PaddleOCR(
                use_angle_cls=True,
                lang="ch",
                cpu_threads=4,
                enable_mkldnn=False,
            )
        logger.info("[OCRRouter] PP-OCRv4 初始化完成")
    return _ppocr_instance


def _run_ppocr(file_path: str, doc_type: str = "unknown", **_ignored) -> list[Document]:
    """PP-OCRv4：适合印刷清晰的营业执照、发票、文档、表格。"""
    import re
    ocr = _get_ppocr()

    try:
        results = list(ocr.predict(file_path))
    except Exception as exc:
        logger.error("[OCRRouter] PP-OCRv4 失败: %s", exc, exc_info=True)
        raise

    # 提取文本
    all_lines: list[tuple[float, str]] = []  # (y_center, text)
    for res in results:
        data = res if isinstance(res, dict) else (res.json.get("res", {}) if hasattr(res, "json") else {})
        texts  = data.get("rec_texts",  [])
        polys  = data.get("dt_polys",   [])
        scores = data.get("rec_scores", [1.0] * len(texts))

        for i, (text, score) in enumerate(zip(texts, scores)):
            if not text.strip() or float(score) < 0.5:
                continue
            # 取 y 中心用于排序（保持阅读顺序）
            y_center = 0.0
            if polys and i < len(polys):
                ys = [p[1] for p in polys[i]]
                y_center = sum(ys) / len(ys)
            all_lines.append((y_center, text.strip()))

    # 按 y 排序，拼成文本
    all_lines.sort(key=lambda x: x[0])
    text = "\n".join(t for _, t in all_lines)

    if not text.strip():
        logger.warning("[OCRRouter] PP-OCRv4 未识别到内容: %s", file_path)

    from models.document import Document
    return [Document(
        page_content=text.strip(),
        metadata={
            "source":   file_path,
            "doc_type": doc_type,
            "label":    "ocr",
            "engine":   "ppocr",
        },
    )]


# ---------------------------------------------------------------------------
# GOT-OCR2 引擎（占位，待部署后填充）
# ---------------------------------------------------------------------------

def _run_got_ocr(file_path: str, **_ignored) -> list[Document]:
    """
    GOT-OCR2：适合设备铭牌、手机拍照、光线复杂场景。
    需要先部署模型：
        python3 -c "
        from transformers import AutoTokenizer, AutoModelForCausalLM
        AutoTokenizer.from_pretrained('ucaslcl/GOT-OCR2_0', cache_dir='/root/models/got-ocr2')
        AutoModelForCausalLM.from_pretrained('ucaslcl/GOT-OCR2_0', cache_dir='/root/models/got-ocr2')
        "
    部署完后设置环境变量：GOT_OCR_MODEL=/root/models/got-ocr2
    """
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
            metadata={
                "source":   file_path,
                "label":    "ocr",
                "engine":   "got_ocr2",
            },
        )]
    except Exception as exc:
        logger.error("[OCRRouter] GOT-OCR2 失败，降级到 PaddleOCR-VL: %s", exc)
        return _run_paddleocr_vl(file_path)


# ---------------------------------------------------------------------------
# PaddleOCR-VL 引擎（现有，兜底）
# ---------------------------------------------------------------------------

def _run_paddleocr_vl(file_path: str, **kwargs) -> list[Document]:
    """PaddleOCR-VL：当前兜底引擎，处理 unknown 和降级场景。"""
    from extractor.image_extractor import ImageExtractor
    extractor = ImageExtractor(file_path, **kwargs)
    return extractor.extract()
    