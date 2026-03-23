"""
extractor/pdf_extractor.py — PDF 提取器。

自动区分两种 PDF：
  文本 PDF（Word 导出、数字原生）→ pypdfium2 直接提取文字层，速度快
  扫描件 PDF（图片扫描进去）    → 转图片 → OCR（PaddleOCR-VL）

判断逻辑：
  每页提取文字，有效字符数 < TEXT_THRESHOLD 视为扫描页
  整个 PDF 超过 SCAN_RATIO 比例的页都是扫描页 → 走 OCR 路径
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import pypdfium2

from .base import BaseExtractor
from models.document import Document

logger = logging.getLogger(__name__)

# 每页有效字符数低于此值视为扫描页（空格/换行不算）
TEXT_THRESHOLD = 30
# 超过此比例的页是扫描页，整个 PDF 走 OCR
SCAN_RATIO = 0.5


class PdfExtractor(BaseExtractor):

    def __init__(
        self,
        file_path: str,
        extract_images: bool = False,
        doc_type: str = "document",
        # OCR 参数（扫描件时透传给 VL）
        vl_rec_backend: str | None = None,
        vl_rec_server_url: str | None = None,
        device: str | None = None,
        **_ignored,
    ):
        self._file_path = file_path
        self._extract_images = extract_images
        self._doc_type = doc_type
        self._vl_kwargs = {k: v for k, v in {
            "vl_rec_backend":    vl_rec_backend,
            "vl_rec_server_url": vl_rec_server_url,
            "device":            device,
        }.items() if v is not None}

    def extract(self) -> list[Document]:
        pdf = pypdfium2.PdfDocument(self._file_path, autoclose=True)
        try:
            pages_text = self._extract_all_text(pdf)
        finally:
            pdf.close()

        total = len(pages_text)
        scan_pages = sum(1 for t in pages_text if len(t.strip()) < TEXT_THRESHOLD)
        scan_ratio = scan_pages / total if total else 0

        logger.info("[PDF] %s  总页数=%d  扫描页=%d  扫描比例=%.0f%%",
                    Path(self._file_path).name, total, scan_pages, scan_ratio * 100)

        if scan_ratio >= SCAN_RATIO:
            logger.info("[PDF] 判定为扫描件，转图片走 OCR")
            return self._ocr_fallback()

        # 文本 PDF：直接返回文字层
        docs = []
        for page_num, text in enumerate(pages_text):
            if text.strip():
                docs.append(Document(
                    page_content=text,
                    metadata={"source": self._file_path, "page": page_num, "label": "page"},
                ))
        logger.info("[PDF] 文本 PDF，提取 %d 页", len(docs))
        return docs

    def _extract_all_text(self, pdf: pypdfium2.PdfDocument) -> list[str]:
        texts = []
        for page in pdf:
            tp = page.get_textpage()
            texts.append(tp.get_text_range())
            tp.close()
            page.close()
        return texts

    def _ocr_fallback(self) -> list[Document]:
        """将 PDF 每页渲染为图片，逐页走 PaddleOCR-VL。"""
        try:
            import pypdfium2 as pdfium
        except ImportError:
            raise ImportError("pypdfium2 未安装")

        docs: list[Document] = []
        pdf = pdfium.PdfDocument(self._file_path, autoclose=True)

        try:
            for page_num, page in enumerate(pdf):
                # 渲染为 150 DPI 图片（平衡质量与速度）
                bitmap = page.render(scale=150/72)
                pil_img = bitmap.to_pil()
                page.close()

                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                    tmp_path = f.name
                try:
                    pil_img.save(tmp_path, "PNG")
                    page_docs = self._ocr_image(tmp_path, page_num)
                    docs.extend(page_docs)
                finally:
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass

        finally:
            pdf.close()

        logger.info("[PDF] 扫描件 OCR 完成，共 %d 页 %d 块", len(set(d.metadata.get('page') for d in docs)), len(docs))
        return docs

    def _ocr_image(self, img_path: str, page_num: int) -> list[Document]:
        from extractor.ocr_router import route_ocr
        from config.settings import settings

        vl_kwargs = {}
        if settings.vl_backend:
            vl_kwargs["vl_rec_backend"] = settings.vl_backend
        if settings.vl_server_url:
            vl_kwargs["vl_rec_server_url"] = settings.vl_server_url

        try:
            page_docs = route_ocr(img_path, doc_type=self._doc_type, **vl_kwargs)
        except Exception as exc:
            logger.error("[PDF] 第 %d 页 OCR 失败: %s", page_num, exc)
            return []

        for doc in page_docs:
            doc.metadata["source"] = self._file_path
            doc.metadata["page"] = page_num

        return page_docs