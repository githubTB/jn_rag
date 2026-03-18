"""
ImageExtractor — extract text from image files using local Tesseract OCR.

Supported formats: JPEG, PNG, GIF, WEBP, BMP, TIFF

Dependencies
------------
    pip install pytesseract Pillow
    # system: apt install tesseract-ocr tesseract-ocr-chi-sim

Usage
-----
    extractor = ImageExtractor("photo.png")
    docs = extractor.extract()

    # Custom language (default: chi_sim+eng)
    extractor = ImageExtractor("scan.jpg", ocr_lang="eng")

    # Custom pre-processing (e.g. force greyscale)
    extractor = ImageExtractor("scan.jpg", preprocess="grey")
"""

from __future__ import annotations

import logging
from pathlib import Path

from .base import BaseExtractor
from models.document import Document

logger = logging.getLogger(__name__)

# Extension → MIME type
_MIME_MAP: dict[str, str] = {
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png":  "image/png",
    ".gif":  "image/gif",
    ".webp": "image/webp",
    ".bmp":  "image/bmp",
    ".tiff": "image/tiff",
    ".tif":  "image/tiff",
}


class ImageExtractor(BaseExtractor):
    """
    OCR-based image extractor using Tesseract via pytesseract.

    Parameters
    ----------
    file_path : str
        Path to the image file.
    ocr_lang : str
        Tesseract language string.
        Single language  : "eng", "chi_sim", "chi_tra"
        Multiple combined: "chi_sim+eng"  (default)
    preprocess : str | None
        Optional image pre-processing before OCR.
        "grey"    — convert to greyscale (improves most scanned docs)
        "thresh"  — greyscale + binary threshold (good for low-contrast images)
        None      — no pre-processing (default)
    psm : int
        Tesseract page segmentation mode (--psm N).
        3  = fully automatic (default)
        6  = single uniform block of text
        11 = sparse text (good for scattered labels)
    """

    def __init__(
        self,
        file_path: str,
        ocr_lang: str = "chi_sim+eng",
        preprocess: str | None = None,
        psm: int = 3,
    ):
        self._file_path = file_path
        self._ocr_lang = ocr_lang
        self._preprocess = preprocess
        self._psm = psm

    # ------------------------------------------------------------------ #
    #  Public                                                              #
    # ------------------------------------------------------------------ #

    def extract(self) -> list[Document]:
        path = Path(self._file_path)
        ext = path.suffix.lower()

        if ext not in _MIME_MAP:
            raise ValueError(
                f"Unsupported image extension: {ext!r}. "
                f"Supported: {', '.join(sorted(_MIME_MAP))}"
            )

        text = self._run_ocr(path)

        return [
            Document(
                page_content=text.strip(),
                metadata={
                    "source": self._file_path,
                    "mime_type": _MIME_MAP[ext],
                    "ocr_lang": self._ocr_lang,
                },
            )
        ]

    # ------------------------------------------------------------------ #
    #  Internal                                                            #
    # ------------------------------------------------------------------ #

    def _run_ocr(self, path: Path) -> str:
        try:
            import pytesseract
            from PIL import Image
        except ImportError as exc:
            raise ImportError(
                "pytesseract and Pillow are required for image extraction.\n"
                "  pip install pytesseract Pillow\n"
                "  # system: apt install tesseract-ocr"
            ) from exc

        img = Image.open(path)

        # Normalise mode — Tesseract works best with RGB or L
        if img.mode == "RGBA":
            # Composite onto white background to drop alpha channel
            background = Image.new("RGB", img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])
            img = background
        elif img.mode == "P":
            img = img.convert("RGB")
        elif img.mode not in ("RGB", "L"):
            img = img.convert("RGB")

        # Optional pre-processing
        if self._preprocess in ("grey", "thresh"):
            img = img.convert("L")          # greyscale
        if self._preprocess == "thresh":
            import PIL.ImageOps
            img = PIL.ImageOps.autocontrast(img)

        config = f"--psm {self._psm}"
        text: str = pytesseract.image_to_string(img, lang=self._ocr_lang, config=config)
        return text
