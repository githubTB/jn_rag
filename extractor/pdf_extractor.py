import io
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pypdfium2
import pypdfium2.raw as pdfium_c

from .base import BaseExtractor
from models.document import Document


class PdfExtractor(BaseExtractor):
    """
    Extract text (and optionally image placeholders) from PDF files.
    Images are noted as [IMAGE] markers without requiring external storage.
    """

    def __init__(self, file_path: str, extract_images: bool = False):
        self._file_path = file_path
        self._extract_images = extract_images

    def extract(self) -> list[Document]:
        return list(self._iter_pages())

    def _iter_pages(self) -> Iterator[Document]:
        pdf = pypdfium2.PdfDocument(self._file_path, autoclose=True)
        try:
            for page_num, page in enumerate(pdf):
                text_page = page.get_textpage()
                content = text_page.get_text_range()
                text_page.close()

                if self._extract_images:
                    image_markers = self._count_images(page)
                    if image_markers:
                        content += f"\n{image_markers}"

                page.close()
                yield Document(
                    page_content=content,
                    metadata={"source": self._file_path, "page": page_num},
                )
        finally:
            pdf.close()

    def _count_images(self, page) -> str:
        try:
            objects = list(page.get_objects(filter=(pdfium_c.FPDF_PAGEOBJ_IMAGE,)))
            count = len(objects)
            return f"[{count} image(s) on this page]" if count else ""
        except Exception:
            return ""
