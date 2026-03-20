"""
ImageExtractor — extract text from image files using PaddleOCR.

Supported formats: JPEG, PNG, GIF, WEBP, BMP, TIFF

Dependencies
------------
    pip install paddlepaddle paddleocr Pillow
"""

from __future__ import annotations

import logging
import os
import tempfile
import uuid
from pathlib import Path

from .base import BaseExtractor
from models.document import Document

logger = logging.getLogger(__name__)

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

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

_ocr_instance = None


def _get_ocr(lang: str):
    global _ocr_instance
    if _ocr_instance is None:
        from paddleocr import PaddleOCR
        try:
            _ocr_instance = PaddleOCR(
                use_textline_orientation=True,
                lang=lang,
                ocr_version="PP-OCRv4",
                cpu_threads=4,
                enable_mkldnn=False,
                det_db_thresh=0.1,  # 进一步降低检测阈值，提高小字体检测率
                det_db_box_thresh=0.3,
                det_db_unclip_ratio=2.0,  # 进一步扩大检测框，提高小字体检测率
                det_limit_side_len=5000,  # 增加检测边长限制，适应表格
                rec_batch_num=1,
            )
        except TypeError:
            _ocr_instance = PaddleOCR(
                use_angle_cls=True,
                lang=lang,
                cpu_threads=4,
                enable_mkldnn=False,
                det_db_thresh=0.1,
                det_db_box_thresh=0.3,
                det_db_unclip_ratio=2.0,
                det_limit_side_len=5000,
            )
    return _ocr_instance


class ImageExtractor(BaseExtractor):
    """
    PaddleOCR-based image extractor.

    Parameters
    ----------
    file_path : str
    lang : str          "ch"（默认）/ "en"
    score_threshold : float   置信度过滤，默认 0.5
    max_side : int            长边像素上限，默认 1500
    """

    def __init__(
        self,
        file_path: str,
        lang: str = "ch",
        score_threshold: float = 0.5,
        max_side: int = 1500,
    ):
        self._file_path = file_path
        self._lang = lang
        self._score_threshold = score_threshold
        self._max_side = max_side

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
        tmp = self._prepare_image(path)
        try:
            text = self._run_ocr(tmp)
        finally:
            if tmp != path and tmp.exists():
                try:
                    tmp.unlink()
                except Exception:
                    pass
        return [Document(
            page_content=text.strip(),
            metadata={"source": self._file_path, "mime_type": _MIME_MAP[ext], "ocr_lang": self._lang},
        )]

    # ------------------------------------------------------------------ #
    #  OCR                                                                 #
    # ------------------------------------------------------------------ #

    def _run_ocr(self, path: Path) -> str:
        try:
            from paddleocr import PaddleOCR  # noqa
        except ImportError as exc:
            raise ImportError("pip install paddlepaddle paddleocr Pillow") from exc

        results = list(_get_ocr(self._lang).predict(str(path)))
    
        import json
        import numpy as np
        
        # 转换 ndarray 和 Font 类型为可序列化的类型
        def convert_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif hasattr(obj, '__class__') and obj.__class__.__name__ == 'Font':
                return "<Font object>"
            elif isinstance(obj, dict):
                return {k: convert_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_serializable(item) for item in obj]
            else:
                return obj
        
        logger.debug("RAW OCR RESULTS:")
        logger.debug(json.dumps(convert_serializable(results), indent=2, ensure_ascii=False))

        # 尝试使用 PaddleOCR 识别
        text = self._rebuild_from_boxes(results)
        
        # 尝试使用 Tesseract OCR 作为补充，特别是对于表格中的月份数据
        logger.info("尝试使用 Tesseract OCR 补充识别")
        try:
            import pytesseract
            from PIL import Image
            
            img = Image.open(path)
            # 对图片进行预处理，增强对比度
            from PIL import ImageEnhance
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.5)
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(1.2)
            
            # 使用 Tesseract OCR 识别
            tesseract_text = pytesseract.image_to_string(img, lang='chi_sim+eng', config='--psm 6')
            logger.debug("TESSERACT OCR RESULTS:")
            logger.debug(tesseract_text)
            
            # 检查 Tesseract 识别结果是否包含月份数据
            has_month_data = any(month in tesseract_text for month in ["1月", "2月", "3月", "4月", "5月", "6月", "7月", "8月", "9月", "10月", "11月", "12月"])
            
            # 如果 Tesseract 识别结果包含月份数据，使用 Tesseract 的结果
            if has_month_data:
                logger.info("Tesseract OCR 识别到月份数据，使用 Tesseract 的结果")
                text = tesseract_text
        except ImportError:
            logger.warning("pytesseract 未安装，无法使用 Tesseract OCR 作为补充")
        except Exception as e:
            logger.warning("使用 Tesseract OCR 时出错: %s", e)

        return text

    # ------------------------------------------------------------------ #
    #  坐标对齐重建表格（移植自 parser.py _rebuild_table_from_boxes）      #
    # ------------------------------------------------------------------ #

    def _rebuild_from_boxes(self, results) -> str:
        """
        用 OCR box 坐标重建表格结构：
          1. 按 Y 中心 + 绝对像素容差分行
          2. 聚类列中心，补全空单元格为 -
          3. 每行按 X 排序后拼接
        兼容新版（dt_polys）和旧版（rec_boxes）两种坐标格式。
        """
        all_items: list[tuple] = []  # (x1,y1,x2,y2,text)

        for res in results:
            try:
                # 新版 predict() 返回 dict
                if isinstance(res, dict):
                    data = res
                else:
                    # 兼容旧版 OCRResult 对象
                    data = res.json.get("res", {}) if hasattr(res, "json") else {}

                texts  = data.get("rec_texts", [])
                polys  = data.get("dt_polys",  [])
                boxes  = data.get("rec_boxes", [])
                scores = data.get("rec_scores", [1.0] * len(texts))

                if not texts:
                    continue

                if polys and len(polys) == len(texts):
                    # 新版 dt_polys: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
                    for poly, text, score in zip(polys, texts, scores):
                        if not text.strip() or float(score) < self._score_threshold:
                            continue
                        xs = [p[0] for p in poly]
                        ys = [p[1] for p in poly]
                        all_items.append((min(xs), min(ys), max(xs), max(ys), text.strip()))
                        logger.debug("OCR BLOCK: %s (%d,%d,%d,%d)", text.strip(), min(xs), min(ys), max(xs), max(ys))
                elif boxes and len(boxes) == len(texts):
                    # 旧版 rec_boxes: [x1,y1,x2,y2]
                    for box, text, score in zip(boxes, texts, scores):
                        if not text.strip() or float(score) < self._score_threshold:
                            continue
                        all_items.append((box[0], box[1], box[2], box[3], text.strip()))
                else:
                    # 无坐标，直接平铺
                    return "\n".join(t for t, s in zip(texts, scores)
                                     if t.strip() and float(s) >= self._score_threshold)
            except Exception as e:
                logger.warning("box 解析失败: %s", e)

        if not all_items:
            return ""

        # 1. 按 Y 中心分行，容差 = max(平均行高 * 0.6, 10)
        avg_h = sum(item[3] - item[1] for item in all_items) / len(all_items)
        tol   = max(avg_h * 0.6, 10)

        all_items.sort(key=lambda x: (x[1] + x[3]) / 2)
        rows: list[list[tuple]] = []
        current_row = [all_items[0]]

        def _row_y(row):
            """行内所有块 y 中心的均值，避免单个块偏移导致跨行判断失误"""
            return sum((it[1] + it[3]) / 2 for it in row) / len(row)

        for item in all_items[1:]:
            y_center = (item[1] + item[3]) / 2
            if abs(y_center - _row_y(current_row)) <= tol:
                current_row.append(item)
            else:
                rows.append(sorted(current_row, key=lambda x: x[0]))
                current_row = [item]
        rows.append(sorted(current_row, key=lambda x: x[0]))

        if len(rows) < 2:
            return "\n".join(" ".join(item[4] for item in row) for row in rows)

        # 2. 聚类列中心
        # 用中位数宽度而非平均值，避免跨列标题（水/电/天然气）干扰列间距计算
        all_x_centers = sorted((item[0] + item[2]) / 2
                                for row in rows for item in row)
        all_widths = sorted(item[2] - item[0] for row in rows for item in row)
        median_w = all_widths[len(all_widths) // 2]
        col_gap  = max(median_w * 0.8, 15)

        col_centers = [all_x_centers[0]]
        for x in all_x_centers[1:]:
            if x - col_centers[-1] > col_gap:
                col_centers.append(x)
            else:
                col_centers[-1] = (col_centers[-1] + x) / 2

        n_cols = len(col_centers)

        def assign_col(xc: float) -> int:
            return min(range(n_cols), key=lambda i: abs(col_centers[i] - xc))

        # 3. 每行分配列，空列填 -
        lines = []
        for row in rows:
            cells = ["-"] * n_cols
            for item in row:
                col_idx = assign_col((item[0] + item[2]) / 2)
                cells[col_idx] = item[4]
            lines.append("  ".join(cells))

        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    #  调试用：返回原始块坐标（供 debug_ocr.py 使用）                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _normalize_box(box) -> tuple[float, float, float, float]:
        if len(box) == 4 and not hasattr(box[0], "__len__"):
            return float(box[0]), float(box[1]), float(box[2]), float(box[3])
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        return float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))

    # ------------------------------------------------------------------ #
    #  Image preprocessing                                                 #
    # ------------------------------------------------------------------ #

    def _prepare_image(self, path: Path) -> Path:
        from PIL import Image, ImageOps, ImageEnhance

        img = Image.open(path)
        try:
            img = ImageOps.exif_transpose(img)
        except Exception:
            pass

        changed = False
        w, h = img.size
        
        # 改进的缩放策略：同时考虑宽高比和最大边长
        if max(w, h) > self._max_side:
            # 计算宽高比
            aspect_ratio = w / h if w > h else h / w
            
            # 对于宽高比大于 2:1 的图片，采用更保守的缩放策略
            if aspect_ratio > 2:
                # 对于宽图，限制宽度不超过 max_side
                if w > h:
                    scale = min(self._max_side / w, 1.0)
                # 对于高图，限制高度不超过 max_side
                else:
                    scale = min(self._max_side / h, 1.0)
            else:
                # 对于普通比例图片，使用原来的长边限制
                scale = self._max_side / max(w, h)
            
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            img = img.resize((new_w, new_h), Image.LANCZOS)
            logger.debug("Resized %s: %dx%d → %dx%d (aspect ratio: %.2f)", path.name, w, h, new_w, new_h, aspect_ratio)
            changed = True

        # 增强对比度和亮度，提高文字识别率
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(2.5)  # 进一步增加对比度
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(1.4)  # 进一步增加亮度
        
        # 锐化处理，提高文字边缘清晰度
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(1.8)  # 增加锐化程度
        
        # 对图片进行局部处理，增强左侧区域的对比度和亮度
        w, h = img.size
        left_region = img.crop((0, 0, w//3, h))  # 左侧 1/3 区域
        enhancer = ImageEnhance.Contrast(left_region)
        left_region = enhancer.enhance(2.0)  # 增强左侧区域的对比度
        enhancer = ImageEnhance.Brightness(left_region)
        left_region = enhancer.enhance(1.2)  # 增强左侧区域的亮度
        enhancer = ImageEnhance.Sharpness(left_region)
        left_region = enhancer.enhance(2.0)  # 增强左侧区域的锐度
        img.paste(left_region, (0, 0))
        
        changed = True

        if not changed and path.suffix.lower() in (".jpg", ".jpeg"):
            return path

        tmp = Path(tempfile.gettempdir()) / f"_ocr_{uuid.uuid4().hex}.jpg"
        img.save(tmp, "JPEG", quality=92)
        return tmp