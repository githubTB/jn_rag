"""
ExtractProcessor — dispatch file extraction by extension.

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

    # Image: Tesseract language
    ExtractProcessor.extract("scan.jpg", ocr_lang="eng")

    # Image: pre-processing before OCR
    ExtractProcessor.extract("photo.png", preprocess="grey")   # greyscale
    ExtractProcessor.extract("photo.png", preprocess="thresh") # threshold

    # Image: Tesseract page-segmentation mode
    ExtractProcessor.extract("labels.png", psm=11)
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
    # Documents
    ".pdf":      PdfExtractor,
    ".docx":     WordExtractor,
    ".docm":     WordExtractor,
    # Presentations
    ".pptx":     PptxExtractor,
    # Images (Tesseract OCR)
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
    """Dispatch file extraction to the correct extractor by file extension."""

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
        # Forward only kwargs the extractor's __init__ accepts
        accepted = set(inspect.signature(extractor_cls.__init__).parameters) - {"self"}
        return extractor_cls(file_path, **{k: v for k, v in kwargs.items() if k in accepted})