# extractor/__init__.py
__all__ = [
    "BaseExtractor",
    "CSVExtractor",
    "ExcelExtractor",
    "HtmlExtractor",
    "ImageExtractor",
    "MarkdownExtractor",
    "PdfExtractor",
    "PptxExtractor",
    "TextExtractor",
    "WordExtractor",
]

# 懒加载：用到哪个才导入哪个
def __getattr__(name):
    if name == "BaseExtractor":
        from .base import BaseExtractor
        return BaseExtractor
    elif name == "CSVExtractor":
        from .csv_extractor import CSVExtractor
        return CSVExtractor
    elif name == "ExcelExtractor":
        from .excel_extractor import ExcelExtractor
        return ExcelExtractor
    elif name == "HtmlExtractor":
        from .html_extractor import HtmlExtractor
        return HtmlExtractor
    elif name == "ImageExtractor":
        from .image_extractor import ImageExtractor
        return ImageExtractor
    elif name == "MarkdownExtractor":
        from .markdown_extractor import MarkdownExtractor
        return MarkdownExtractor
    elif name == "PdfExtractor":
        from .pdf_extractor import PdfExtractor
        return PdfExtractor
    elif name == "PptxExtractor":
        from .pptx_extractor import PptxExtractor
        return PptxExtractor
    elif name == "TextExtractor":
        from .text_extractor import TextExtractor
        return TextExtractor
    elif name == "WordExtractor":
        from .word_extractor import WordExtractor
        return WordExtractor
    raise AttributeError(f"模块 {__name__!r} 没有属性 {name!r}")