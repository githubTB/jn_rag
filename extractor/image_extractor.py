"""
extractor/image_extractor.py — PaddleOCR-VL-1.5 图片文字提取器。

推理后端选择
-----------
macOS（Apple Silicon）：
    ExtractProcessor.extract("img.jpg", device="cpu")

Linux GPU（vLLM 服务，推荐生产）：
    先启动服务：
        paddleocr genai_server --model_name PaddleOCR-VL-1.5-0.9B \\
            --backend vllm --port 8118
    再调用：
        ExtractProcessor.extract("img.jpg",
            vl_rec_backend="vllm-server",
            vl_rec_server_url="http://localhost:8118/v1")

.env 配置（优先级低于代码传参）：
    VL_BACKEND=vllm-server
    VL_SERVER_URL=http://<ip>:8118/v1
    VL_DEVICE=
"""

from __future__ import annotations

import gc
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .base import BaseExtractor
from models.document import Document

logger = logging.getLogger(__name__)

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

# ---------------------------------------------------------------------------
# 支持的 MIME 类型
# ---------------------------------------------------------------------------

_MIME_MAP: dict[str, str] = {
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png":  "image/png",
    ".bmp":  "image/bmp",
    ".webp": "image/webp",
    ".tiff": "image/tiff",
    ".tif":  "image/tiff",
}

# ---------------------------------------------------------------------------
# 环境变量读取
# ---------------------------------------------------------------------------

def _env(key: str, default: str | None = None) -> str | None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    return os.environ.get(key, default)


# ---------------------------------------------------------------------------
# PaddleOCRVL 单例（进程内复用）
# ---------------------------------------------------------------------------

_pipeline: Any | None = None


def _get_pipeline(**init_kwargs) -> Any:
    global _pipeline
    if _pipeline is None:
        try:
            from paddleocr import PaddleOCRVL
        except ImportError as exc:
            raise ImportError(
                'PaddleOCR-VL 未安装: pip install "paddleocr[doc-parser]"'
            ) from exc
        logger.info("[VL] 初始化 PaddleOCRVL: %s", init_kwargs)
        _pipeline = PaddleOCRVL(**init_kwargs)
        logger.info("[VL] 初始化完成")
    return _pipeline


def reset_pipeline() -> None:
    """强制下次调用时重新初始化（切换后端或测试用）。"""
    global _pipeline
    _pipeline = None
    logger.info("[VL] pipeline 已重置")


# ---------------------------------------------------------------------------
# 主类
# ---------------------------------------------------------------------------

class ImageExtractor(BaseExtractor):
    """
    基于 PaddleOCR-VL-1.5 的图片文字提取器。

    Parameters
    ----------
    file_path              : 图片路径（本地或 HTTP URL）
    max_file_mb            : 本地文件大小上限，默认 50 MB
    use_layout_detection   : 是否启用布局检测（None=自动，True=强制开启）
    use_doc_orientation_classify : 是否自动纠正文档方向，默认 False
    use_doc_unwarping      : 是否启用文档矫正展平，默认 False
    vl_rec_backend         : 推理后端（None=自动 / "vllm-server" 等）
    vl_rec_server_url      : 外部推理服务地址
    vl_rec_max_concurrency : 并发请求数，默认 1
    device                 : 推理设备（None=自动 / "cpu" / "gpu:0"）
    output_format          : "markdown"（默认）或 "text"
    """

    def __init__(
        self,
        file_path: str,
        *,
        doc_type: str = "unknown", 
        max_file_mb: float = 50.0,
        use_layout_detection: bool | None = None,
        use_doc_orientation_classify: bool = False,
        use_doc_unwarping: bool = False,
        vl_rec_backend: str | None = None,
        vl_rec_server_url: str | None = None,
        vl_rec_max_concurrency: int = 1,
        device: str | None = None,
        output_format: str = "markdown",
        release_pipeline_after_extract: bool | None = None,
        max_pixels: int | None = None,  # 废弃参数，静默忽略
        **_ignored,
    ):
        self._file_path = file_path
        self._doc_type = doc_type
        self._max_file_mb = max_file_mb
        self._use_layout_detection = use_layout_detection
        self._use_doc_orientation_classify = use_doc_orientation_classify
        self._use_doc_unwarping = use_doc_unwarping
        self._doc_type = doc_type
        self._vl_rec_backend    = vl_rec_backend    or _env("VL_BACKEND")
        self._vl_rec_server_url = vl_rec_server_url or _env("VL_SERVER_URL")
        self._vl_rec_max_concurrency = vl_rec_max_concurrency
        self._device = device or _env("VL_DEVICE")
        self._output_format = output_format
        self._release_pipeline_after_extract = release_pipeline_after_extract

        logger.debug(
            "[VL] 配置: backend=%s url=%s device=%s",
            self._vl_rec_backend, self._vl_rec_server_url, self._device,
        )

    # ------------------------------------------------------------------
    #  Public
    # ------------------------------------------------------------------

    def extract(self) -> list[Document]:
        path = Path(self._file_path)
        is_url = self._file_path.startswith(("http://", "https://"))
        ext = path.suffix.lower()

        # ── doc_type 路由：PP-OCRv4 / GOT-OCR2 / PaddleOCR-VL ────────
        # license/invoice/table/document → PP-OCRv4（轻量，CPU）
        # nameplate → GOT-OCR2（复杂场景，GPU）
        # unknown → PaddleOCR-VL 兜底
        if not is_url and self._doc_type != "unknown":
            try:
                from extractor.ocr_router import route_ocr
                logger.info("[VL] doc_type=%s，走 OCR 路由", self._doc_type)
                return route_ocr(
                    self._file_path,
                    doc_type=self._doc_type,
                    vl_rec_backend=self._vl_rec_backend,
                    vl_rec_server_url=self._vl_rec_server_url,
                    device=self._device,
                )
            except Exception as exc:
                logger.warning("[VL] OCR 路由失败，降级到 PaddleOCR-VL: %s", exc)
                # 降级继续走下面的 PaddleOCR-VL 流程

        infer_path: str = self._file_path
        tmp_path: Path | None = None

        if not is_url:
            self._validate_image(path, ext)
            tmp_path = self._preprocess_image(path)
            if tmp_path is not None:
                infer_path = str(tmp_path)
                logger.info("[VL] 预处理完成: %s", tmp_path.name)

        logger.info("[VL] 开始推理: %s", Path(infer_path).name)
        try:
            results = self._run_pipeline(infer_path)
        finally:
            if tmp_path is not None and tmp_path.exists():
                try:
                    tmp_path.unlink()
                except Exception:
                    pass
            self._maybe_release_pipeline()

        logger.info("[VL] 推理完成，共 %d 块", len(results))

        mime = _MIME_MAP.get(ext, "image/jpeg")
        docs = []
        for idx, (text, meta) in enumerate(results):
            logger.debug("[VL] 块[%d] label=%s 字符=%d", idx, meta.get("label"), len(text))
            docs.append(Document(
                page_content=text,
                metadata={"source": self._file_path, "mime_type": mime,
                          "block_index": idx, **meta},
            ))

        if not docs:
            logger.warning("[VL] 未识别到任何内容: %s", self._file_path)
            docs = [Document(page_content="",
                             metadata={"source": self._file_path, "mime_type": mime})]

        total_chars = sum(len(d.page_content) for d in docs)
        logger.info("[VL] 输出共 %d 字符", total_chars)
        return docs

    # ------------------------------------------------------------------
    #  校验
    # ------------------------------------------------------------------

    _MAGIC: list[tuple[bytes, str]] = [
        (b"\xff\xd8\xff",        "image/jpeg"),
        (b"\x89PNG\r\n\x1a\n",  "image/png"),
        (b"BM",                  "image/bmp"),
        (b"RIFF",                "image/webp"),
        (b"II\x2a\x00",         "image/tiff"),
        (b"MM\x00\x2a",         "image/tiff"),
        (b"GIF87a",              "image/gif"),
        (b"GIF89a",              "image/gif"),
    ]

    def _validate_image(self, path: Path, ext: str) -> None:
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {path}")
        file_mb = path.stat().st_size / (1024 * 1024)
        if file_mb == 0:
            raise ValueError(f"文件为空: {path}")
        if ext not in _MIME_MAP:
            raise ValueError(f"不支持的扩展名 {ext!r}，支持: {', '.join(sorted(_MIME_MAP))}")
        logger.info("[VL] 文件: %s  大小: %.1f MB", path.name, file_mb)
        if file_mb > self._max_file_mb:
            raise ValueError(f"文件过大 ({file_mb:.1f} MB > {self._max_file_mb} MB): {path}")

        with path.open("rb") as f:
            header = f.read(12)
        detected_mime: str | None = None
        for magic, mime in self._MAGIC:
            if header.startswith(magic):
                detected_mime = mime
                break
        if detected_mime == "image/webp" and header[8:12] != b"WEBP":
            detected_mime = None
        if detected_mime is None:
            raise ValueError(f"文件内容不是有效图片: {path.name}")
        if detected_mime != _MIME_MAP[ext]:
            logger.warning("[VL] 扩展名与实际格式不符 (%s vs %s)，尝试继续", ext, detected_mime)

        # 用 Pillow verify() 轻量校验，只读文件头，不解码整张图
        # 比 cv2.imread 省 10-50x 内存（尤其对 4K+ 原图）
        try:
            from PIL import Image
            with Image.open(path) as probe:
                probe.verify()   # 只读头部，不解码像素
            # verify() 后需重新 open 才能读尺寸
            with Image.open(path) as probe:
                w, h = probe.size
            logger.info("[VL] 图片尺寸: %dx%d px", w, h)
        except ImportError:
            logger.debug("[VL] Pillow 未安装，跳过解码校验")
        except Exception as exc:
            raise ValueError(f"图片文件损坏: {path.name}") from exc

    # ------------------------------------------------------------------
    #  预处理
    # ------------------------------------------------------------------

    def _preprocess_image(self, path: Path) -> Path | None:
        """EXIF 修正 + 超大图降采样，返回临时文件路径或 None（无需处理时）。"""
        import uuid

        try:
            from PIL import Image, ImageOps
        except ImportError:
            logger.debug("[VL] Pillow 未安装，跳过预处理")
            return None

        output_img = None
        changed = False
        try:
            with Image.open(path) as opened:
                img = opened

                # EXIF 方向修正
                try:
                    oriented = ImageOps.exif_transpose(img)
                    if oriented is not img:
                        img = oriented
                        changed = True
                        logger.info("[VL] EXIF 方向已修正: %s", path.name)
                except Exception as e:
                    logger.debug("[VL] EXIF 修正跳过: %s", e)

                # 超大图降采样（>3500px 长边）
                w, h = img.size
                if max(w, h) > 3500:
                    scale = 3500 / max(w, h)
                    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
                    resized = img.resize((nw, nh), Image.LANCZOS)
                    if resized is not img and img is not opened:
                        img.close()
                    img = resized
                    changed = True
                    logger.info("[VL] 降采样: %dx%d → %dx%d", w, h, nw, nh)

                if not changed:
                    if img is not opened:
                        img.close()
                    return None

                output_img = img.copy()
                if img is not opened:
                    img.close()

            suffix = path.suffix.lower() or ".jpg"
            tmp = Path(tempfile.gettempdir()) / f"_vlocr_{uuid.uuid4().hex}{suffix}"
            kwargs: dict = {}
            if suffix in (".jpg", ".jpeg"):
                kwargs = {"quality": 92, "optimize": True}
            output_img.save(tmp, **kwargs)
            return tmp
        finally:
            if output_img is not None:
                output_img.close()

    # ------------------------------------------------------------------
    #  Pipeline
    # ------------------------------------------------------------------

    def _build_init_kwargs(self) -> dict:
        """构建 PaddleOCRVL 初始化参数。"""
        kwargs: dict = {
            "vl_rec_max_concurrency":       self._vl_rec_max_concurrency,
            "use_doc_orientation_classify": self._use_doc_orientation_classify,
            "use_doc_unwarping":            self._use_doc_unwarping,
        }
        if self._vl_rec_backend is not None:
            kwargs["vl_rec_backend"] = self._vl_rec_backend
        if self._vl_rec_server_url:
            kwargs["vl_rec_server_url"] = self._vl_rec_server_url
        if self._device:
            kwargs["device"] = self._device
        if self._use_layout_detection is not None:
            kwargs["use_layout_detection"] = self._use_layout_detection

        logger.info("[VL] pipeline 参数: backend=%s url=%s device=%s doc_type=%s",
                    self._vl_rec_backend, self._vl_rec_server_url, self._device, self._doc_type)
        return kwargs

    def _run_pipeline(self, infer_path: str) -> list[tuple[str, dict]]:
        pipeline = _get_pipeline(**self._build_init_kwargs())
        if not hasattr(pipeline, "predict"):
            raise RuntimeError("PaddleOCRVL 无 predict 方法，请升级: pip install -U paddleocr")

        logger.info("[VL] pipeline.predict: %s", infer_path)
        parsed_results: list[tuple[str, dict]] = []
        result_count = 0
        try:
            for raw_result in pipeline.predict(infer_path):
                result_count += 1
                parsed_results.extend(self._parse_results([raw_result]))
                del raw_result
        except Exception as exc:
            logger.error("[VL] predict 失败: %s", exc, exc_info=True)
            raise
        finally:
            gc.collect()

        logger.info("[VL] predict 返回 %d 个结果", result_count)
        return parsed_results

    def _maybe_release_pipeline(self) -> None:
        if self._release_pipeline_after_extract is None:
            should_release = (
                (self._device or "").lower() == "cpu"
                and not self._vl_rec_backend
                and not self._vl_rec_server_url
            )
        else:
            should_release = self._release_pipeline_after_extract

        if not should_release:
            return

        logger.info("[VL] 当前为本地 CPU 模式，提取后主动释放 pipeline")
        reset_pipeline()
        gc.collect()

    # ------------------------------------------------------------------
    #  结果解析
    # ------------------------------------------------------------------

    def _parse_results(self, raw_results: list) -> list[tuple[str, dict]]:
        output: list[tuple[str, dict]] = []
        for result in raw_results:
            logger.debug("[VL] 解析 result 类型: %s", type(result).__name__)

            blocks = self._extract_blocks(result)
            if blocks:
                output.extend(blocks)
                continue

            md = self._result_to_text(result)
            if md:
                text = md if self._output_format == "markdown" else _strip_markdown(md)
                output.append((text.strip(), {"label": "page"}))

        return output

    def _extract_blocks(self, result) -> list[tuple[str, dict]]:
        """从 parsing_res_list 提取逐块内容。"""
        blocks_out: list[tuple[str, dict]] = []
        try:
            res_dict = dict(result) if hasattr(result, "keys") else {}
            if not res_dict and hasattr(result, "res"):
                res_dict = result.res or {}

            # 主路径：parsing_res_list
            for blk in res_dict.get("parsing_res_list", []):
                if isinstance(blk, dict):
                    label = blk.get("label", "text")
                    text  = (blk.get("content") or blk.get("markdown")
                             or blk.get("text") or "").strip()
                    bbox  = blk.get("bbox", [])
                else:
                    label = getattr(blk, "label", "text")
                    text  = (getattr(blk, "content", "") or getattr(blk, "markdown", "")
                             or getattr(blk, "text", "") or "").strip()
                    bbox  = getattr(blk, "bbox", [])
                if not text:
                    continue
                if self._output_format == "text":
                    text = _strip_markdown(text)
                meta: dict = {"label": label}
                if bbox:
                    meta["bbox"] = bbox
                blocks_out.append((text, meta))

            if blocks_out:
                logger.info("[VL] parsing_res_list 解析到 %d 块", len(blocks_out))
                return blocks_out

            # 兜底：旧版 blocks / ocr_res 字段
            for key in ("blocks", "ocr_res", "rec_res"):
                for blk in res_dict.get(key, []):
                    if not isinstance(blk, dict):
                        continue
                    label = blk.get("label", "text")
                    text  = (blk.get("content") or blk.get("markdown")
                             or blk.get("text") or "").strip()
                    if not text:
                        continue
                    if self._output_format == "text":
                        text = _strip_markdown(text)
                    blocks_out.append((text, {"label": label}))
                if blocks_out:
                    logger.info("[VL] %r 字段解析到 %d 块", key, len(blocks_out))
                    return blocks_out

        except Exception as exc:
            logger.warning("[VL] 块解析失败，降级到整页兜底: %s", exc)
        return blocks_out

    def _result_to_text(self, result) -> str:
        """整页兜底：依次尝试 res 字段 → save_to_json → save_to_markdown。"""
        import json

        # 方式 A：result.res 直接含文本
        res_dict = result.res if hasattr(result, "res") else {}
        if isinstance(res_dict, dict):
            for key in ("markdown", "text", "content", "rec_text", "ocr_text"):
                val = res_dict.get(key, "")
                if val and isinstance(val, str) and val.strip():
                    logger.info("[VL] 从 res[%r] 取到文本 %d 字符", key, len(val))
                    return val

        # 方式 B：save_to_json
        if hasattr(result, "save_to_json"):
            tmp = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
                    tmp = f.name
                result.save_to_json(tmp)
                with open(tmp, encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    for key in ("markdown", "text", "content"):
                        val = data.get(key, "")
                        if val and isinstance(val, str) and val.strip():
                            return val
                    texts = [
                        blk.get("markdown") or blk.get("text") or blk.get("content") or ""
                        for blk in data.get("blocks", [])
                    ]
                    joined = "\n\n".join(t.strip() for t in texts if t.strip())
                    if joined:
                        return joined
            except Exception as exc:
                logger.warning("[VL] save_to_json 失败: %s", exc)
            finally:
                if tmp and os.path.exists(tmp):
                    try:
                        os.unlink(tmp)
                    except Exception:
                        pass

        # 方式 C：save_to_markdown
        if hasattr(result, "save_to_markdown"):
            tmp = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
                    tmp = f.name
                result.save_to_markdown(tmp)
                with open(tmp, encoding="utf-8") as f:
                    return f.read()
            except Exception as exc:
                logger.warning("[VL] save_to_markdown 失败: %s", exc)
            finally:
                if tmp and os.path.exists(tmp):
                    try:
                        os.unlink(tmp)
                    except Exception:
                        pass

        return ""


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _strip_markdown(text: str) -> str:
    """去除常见 Markdown 语法，输出纯文本。"""
    text = re.sub(r"```[^\n]*\n(.*?)```", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*{1,3}(.*?)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,3}(.*?)_{1,3}", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"^\|?[-:| ]+\|?\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
