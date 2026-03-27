"""
ExtractProcessor — route files to fixed-format extractors by suffix.

职责边界
--------
- 本模块只负责“按文件后缀选择解析器”
- 不负责业务文档分类（license / invoice / nameplate 等）
- `pdf` 和 `image` 是特殊解析器：
  - `pdf` 先尝试文字层提取，扫描件再降级到 OCR
  - `image` 直接走 OCR / 版面结构识别

固定格式路由
------------
- `.csv`               -> CSVExtractor
- `.xlsx` / `.xls`     -> ExcelExtractor
- `.docx` / `.docm`    -> WordExtractor
- `.pptx`              -> PptxExtractor
- `.md` / `.html` 等   -> 对应文本类解析器
- 图片后缀             -> ImageExtractor
- `.pdf`               -> PdfExtractor

Quick start
-----------
    from extract_processor import ExtractProcessor

    # Returns list[Document]
    docs = ExtractProcessor.extract("report.pdf")

    # Returns a single concatenated string
    text = ExtractProcessor.extract_text("data.csv")

Per-extractor kwargs (forwarded automatically)
----------------------------------------------
    # CSV: pick a column as the source label
    ExtractProcessor.extract("data.csv", source_column="url")

    # PDF: mark image objects on each page
    ExtractProcessor.extract("report.pdf", extract_images=True)

    # Image / PDF: OCR-related kwargs are forwarded only to special extractors
    ExtractProcessor.extract("scan.jpg", output_format="markdown")
"""

from __future__ import annotations

import inspect
import logging
from pathlib import Path

from extractor import (
    BaseExtractor,
    CSVExtractor,
    ExcelExtractor,
    HtmlExtractor,
    ImageExtractor,
    MarkdownExtractor,
    PdfExtractor,
    PptxExtractor,
    TextExtractor,
    WordExtractor,
)
from models.document import Document

logger = logging.getLogger(__name__)

_EXT_MAP: dict[str, type[BaseExtractor]] = {
    # Plain text
    ".txt":      TextExtractor,
    ".log":      TextExtractor,
    ".json":     TextExtractor,
    # Markup
    ".md":       MarkdownExtractor,
    ".markdown": MarkdownExtractor,
    ".mdx":      MarkdownExtractor,
    ".htm":      HtmlExtractor,
    ".html":     HtmlExtractor,
    # Spreadsheets
    ".csv":      CSVExtractor,
    ".xlsx":     ExcelExtractor,
    ".xls":      ExcelExtractor,
    # Documents / special parsers
    ".pdf":      PdfExtractor,
    ".docx":     WordExtractor,
    ".docm":     WordExtractor,
    # Presentations
    ".pptx":     PptxExtractor,
    # Images (OCR / layout analysis)
    ".jpg":      ImageExtractor,
    ".jpeg":     ImageExtractor,
    ".png":      ImageExtractor,
    ".gif":      ImageExtractor,
    ".webp":     ImageExtractor,
    ".bmp":      ImageExtractor,
    ".tiff":     ImageExtractor,
    ".tif":      ImageExtractor,
}


class ExtractProcessor:
    """Choose the correct extractor from the file suffix."""

    @classmethod
    def extract(cls, file_path: str, **kwargs) -> list[Document]:
        """Extract and return a list of Document objects."""
        return cls._build(file_path, **kwargs).extract()

    @classmethod
    def extract_text(cls, file_path: str, separator: str = "\n\n", **kwargs) -> str:
        """Extract and return a single concatenated string."""
        return separator.join(d.page_content for d in cls.extract(file_path, **kwargs))

    @classmethod
    def supported_extensions(cls) -> list[str]:
        return sorted(_EXT_MAP.keys())

    @classmethod
    def _build(cls, file_path: str, **kwargs) -> BaseExtractor:
        ext = Path(file_path).suffix.lower()
        extractor_cls = _EXT_MAP.get(ext, TextExtractor)
        if ext not in _EXT_MAP:
            logger.warning("Unknown extension %r — falling back to TextExtractor", ext)
        # ExtractProcessor only does suffix routing; kwargs are filtered per extractor.
        accepted = set(inspect.signature(extractor_cls.__init__).parameters) - {"self"}
        return extractor_cls(file_path, **{k: v for k, v in kwargs.items() if k in accepted})
