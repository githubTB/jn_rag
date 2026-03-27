"""
extractor/image_extractor.py — 图片文字提取器。

支持两种后端，通过 .env 切换：
  1. 远程 Ollama GLM-OCR（ollama 服务）

.env 配置：
    # 远程 Ollama GLM-OCR
    VL_BACKEND=ollama
    OLLAMA_BASE_URL=http://localhost:11434
    OLLAMA_OCR_MODEL=glm-ocr:latest
"""

from __future__ import annotations

import base64
import gc
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from core.doc_type_classifier import classify_doc_type
from .base import BaseExtractor
from models.document import Document

logger = logging.getLogger(__name__)

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

_MIME_MAP: dict[str, str] = {
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png":  "image/png",
    ".bmp":  "image/bmp",
    ".webp": "image/webp",
    ".tiff": "image/tiff",
    ".tif":  "image/tiff",
}

def _env(key: str, default: str | None = None) -> str | None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    return os.environ.get(key, default)


# PaddleOCRVL 单例
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
    global _pipeline
    _pipeline = None
    logger.info("[VL] pipeline 已重置")


class ImageExtractor(BaseExtractor):
    """
    图片文字提取器。

    VL_BACKEND=vllm-server → 走远程 PaddleOCR-VL
    VL_BACKEND=ollama      → 走本地 Ollama GLM-OCR
    """

    def __init__(
        self,
        file_path: str,
        *,
        max_file_mb: float = 50.0,
        vl_rec_backend: str | None = None,
        vl_rec_server_url: str | None = None,
        vl_rec_max_concurrency: int = 1,
        output_format: str = "markdown",
        doc_type: str = "unknown",
        use_layout_detection: bool | None = None,
        use_doc_orientation_classify: bool = False,
        use_doc_unwarping: bool = False,
        device: str | None = None,
        release_pipeline_after_extract: bool | None = None,
        max_pixels: int | None = None,
        **_ignored,
    ):
        self._file_path = file_path
        self._max_file_mb = max_file_mb
        self._doc_type = doc_type
        self._output_format = output_format
        self._vl_rec_max_concurrency = vl_rec_max_concurrency

        # 始终从 .env 读取，外部参数只作为兜底
        # 不管从哪里创建（直接/PdfExtractor内部/route_ocr），都能正确读到配置
        self._vl_rec_backend    = _env("VL_BACKEND")    or vl_rec_backend
        self._vl_rec_server_url = _env("VL_SERVER_URL") or vl_rec_server_url

        # Ollama 配置
        self._ollama_base_url  = _env("OLLAMA_BASE_URL",  "http://localhost:11434")
        self._ollama_ocr_model = _env("OLLAMA_OCR_MODEL", "glm-ocr:latest")

        logger.debug("[OCR] backend=%s", self._vl_rec_backend)

    def extract(self) -> list[Document]:
        path = Path(self._file_path)
        ext  = path.suffix.lower()

        self._validate_image(path, ext)

        # 预处理：EXIF 修正 + 超大图降采样
        tmp_path = self._preprocess_image(path)
        infer_path = str(tmp_path) if tmp_path else self._file_path

        logger.info("[OCR] 开始推理: %s  backend=%s", path.name, self._vl_rec_backend)
        try:
            # 根据 backend 分流
            if self._vl_rec_backend == "ollama":
                results = self._run_ollama(infer_path)
            else:
                results = self._run_paddleocr_vl(infer_path)
        finally:
            if tmp_path and tmp_path.exists():
                try:
                    tmp_path.unlink()
                except Exception:
                    pass

        logger.info("[OCR] 推理完成，共 %d 块", len(results))

        mime = _MIME_MAP.get(ext, "image/jpeg")
        docs = []
        for idx, (text, meta) in enumerate(results):
            logger.debug("[OCR] 块[%d] label=%s 字符=%d", idx, meta.get("label"), len(text))
            docs.append(Document(
                page_content=text,
                metadata={"source": self._file_path, "mime_type": mime,
                          "block_index": idx, **meta},
            ))

        if not docs:
            logger.warning("[OCR] 未识别到任何内容: %s", self._file_path)
            docs = [Document(page_content="",
                             metadata={"source": self._file_path, "mime_type": mime})]

        logger.info("[OCR] 输出共 %d 字符", sum(len(d.page_content) for d in docs))
        
        decision = classify_doc_type(docs, file_path=self._file_path)
        for doc in docs:
            doc.metadata["inferred_doc_type"] = decision.doc_type
            doc.metadata["doc_type_confidence"] = decision.confidence
            doc.metadata["doc_type_evidence"] = decision.evidence
        
        return docs

    # ------------------------------------------------------------------
    #  Ollama GLM-OCR 后端
    # ------------------------------------------------------------------

    def _run_ollama(self, infer_path: str) -> list[tuple[str, dict]]:
        """走本地 Ollama GLM-OCR。"""
        import urllib.request
        import json

        # 图片转 base64
        with open(infer_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()

        payload = {
            "model": self._ollama_ocr_model,
            "messages": [
                {
                    "role": "user",
                    "content": "识别图片中的所有文字内容，保持原有格式和结构，表格用竖线分隔。",
                    "images": [img_b64],
                }
            ],
            "stream": False,
        }

        url = f"{self._ollama_base_url.rstrip('/')}/api/chat"
        logger.info("[Ollama] 请求: %s  model=%s", url, self._ollama_ocr_model)

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read())
            text = data.get("message", {}).get("content", "").strip()
            logger.info("[Ollama] 识别完成: %d 字符", len(text))

            # 打印识别内容方便调试
            print(f"\n{'='*60}")
            print(f"[Ollama GLM-OCR] 文件: {Path(infer_path).name}")
            print(f"[Ollama GLM-OCR] 识别字符数: {len(text)}")
            print(f"[Ollama GLM-OCR] 内容:\n{text}")
            print(f"{'='*60}\n")

            return [(text, {"label": "ocr", "engine": "glm-ocr"})] if text else []

        except Exception as exc:
            logger.error("[Ollama] 调用失败: %s", exc, exc_info=True)
            raise

    # ------------------------------------------------------------------
    #  PaddleOCR-VL 远程后端
    # ------------------------------------------------------------------

    def _run_paddleocr_vl(self, infer_path: str) -> list[tuple[str, dict]]:
        """走远程 PaddleOCR-VL（vLLM 服务）。"""
        pipeline = _get_pipeline(**self._build_vl_init_kwargs())
        parsed: list[tuple[str, dict]] = []
        try:
            for raw in pipeline.predict(infer_path):
                parsed.extend(self._parse_vl_result(raw))
                del raw
        except Exception as exc:
            logger.error("[VL] predict 失败: %s", exc, exc_info=True)
            raise
        finally:
            gc.collect()
        return parsed

    def _build_vl_init_kwargs(self) -> dict:
        kwargs: dict = {"vl_rec_max_concurrency": self._vl_rec_max_concurrency}
        if self._vl_rec_backend:
            kwargs["vl_rec_backend"] = self._vl_rec_backend
        if self._vl_rec_server_url:
            kwargs["vl_rec_server_url"] = self._vl_rec_server_url
        logger.info("[VL] 参数: backend=%s url=%s", self._vl_rec_backend, self._vl_rec_server_url)
        return kwargs

    def _parse_vl_result(self, result) -> list[tuple[str, dict]]:
        blocks: list[tuple[str, dict]] = []
        try:
            res_dict = dict(result) if hasattr(result, "keys") else {}
            if not res_dict and hasattr(result, "res"):
                res_dict = result.res or {}

            for blk in res_dict.get("parsing_res_list", []):
                if isinstance(blk, dict):
                    label = blk.get("label", "text")
                    text  = (blk.get("content") or blk.get("markdown") or blk.get("text") or "").strip()
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
                blocks.append((text, meta))

            if blocks:
                return blocks

            for key in ("blocks", "ocr_res", "rec_res"):
                for blk in res_dict.get(key, []):
                    if not isinstance(blk, dict):
                        continue
                    text = (blk.get("content") or blk.get("markdown") or blk.get("text") or "").strip()
                    if not text:
                        continue
                    blocks.append((text, {"label": blk.get("label", "text")}))
                if blocks:
                    return blocks

        except Exception as exc:
            logger.warning("[VL] 块解析失败，走整页兜底: %s", exc)

        md = self._result_to_text(result)
        if md:
            text = md if self._output_format == "markdown" else _strip_markdown(md)
            blocks.append((text.strip(), {"label": "page"}))

        return blocks

    def _result_to_text(self, result) -> str:
        import json

        res_dict = result.res if hasattr(result, "res") else {}
        if isinstance(res_dict, dict):
            for key in ("markdown", "text", "content", "rec_text", "ocr_text"):
                val = res_dict.get(key, "")
                if val and isinstance(val, str) and val.strip():
                    return val

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
                    texts = [blk.get("markdown") or blk.get("text") or "" for blk in data.get("blocks", [])]
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

    # ------------------------------------------------------------------
    #  校验 + 预处理
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
            raise ValueError(f"不支持的扩展名 {ext!r}")
        if file_mb > self._max_file_mb:
            raise ValueError(f"文件过大 ({file_mb:.1f} MB > {self._max_file_mb} MB)")
        logger.info("[OCR] 文件: %s  %.1f MB", path.name, file_mb)

        with path.open("rb") as f:
            header = f.read(12)
        detected = None
        for magic, mime in self._MAGIC:
            if header.startswith(magic):
                detected = mime
                break
        if detected is None:
            raise ValueError(f"文件内容不是有效图片: {path.name}")

    def _preprocess_image(self, path: Path) -> Path | None:
        """EXIF 修正 + 超大图降采样（长边 > 2000px 时压缩）。"""
        import uuid
        try:
            from PIL import Image, ImageOps
        except ImportError:
            return None

        output_img = None
        changed = False
        try:
            with Image.open(path) as opened:
                img = opened

                try:
                    oriented = ImageOps.exif_transpose(img)
                    if oriented is not img:
                        img = oriented
                        changed = True
                        logger.info("[OCR] EXIF 修正: %s", path.name)
                except Exception:
                    pass

                w, h = img.size
                if max(w, h) > 2000:
                    scale = 2000 / max(w, h)
                    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
                    img = img.resize((nw, nh), Image.LANCZOS)
                    changed = True
                    logger.info("[OCR] 降采样: %dx%d → %dx%d", w, h, nw, nh)

                if not changed:
                    return None

                output_img = img.copy()

            suffix = path.suffix.lower() or ".jpg"
            tmp = Path(tempfile.gettempdir()) / f"_ocr_{uuid.uuid4().hex}{suffix}"
            kwargs: dict = {"quality": 85, "optimize": True} if suffix in (".jpg", ".jpeg") else {}
            output_img.save(tmp, **kwargs)
            return tmp
        finally:
            if output_img:
                output_img.close()


def _strip_markdown(text: str) -> str:
    text = re.sub(r"```[^\n]*\n(.*?)```", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*{1,3}(.*?)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,3}(.*?)_{1,3}", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"^\|?[-:| ]+\|?\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
