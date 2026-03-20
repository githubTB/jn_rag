"""
ImageExtractor — PaddleOCR-VL-1.5 图片文字提取器。

架构说明
--------
PaddleOCR-VL-1.5 是端到端 VLM，内部已包含：
  - NaViT 动态分辨率视觉编码器（自动处理任意尺寸，无需手动缩放）
  - ERNIE-4.5-0.3B 语言模型（直接输出结构化文本/表格/公式）
  - PP-DocLayoutV3 布局检测（可选，开启后识别更准）

本模块只负责：
  1. 文件校验（大小限制，格式检查）
  2. 调用 PaddleOCRVL.predict()
  3. 将结果拼成 Document 列表

不做任何手工图片预处理（缩放/裁剪/二值化），VLM 内部已处理。

安装依赖
--------
    # macOS（Apple Silicon，必须用 mlx-vlm-server 后端）
    pip install paddlepaddle==3.2.1 \\
        -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
    pip install "paddleocr[doc-parser]" mlx-vlm

    # Linux GPU（CUDA 12.6）
    pip install paddlepaddle-gpu==3.2.1 \\
        -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
    pip install "paddleocr[doc-parser]"

    # Linux CPU（慢但可用）
    pip install paddlepaddle==3.2.1
    pip install "paddleocr[doc-parser]"

推理后端选择
-----------
macOS（Apple Silicon）— 直接 CPU 推理，无需启动任何服务：
    ImageExtractor("img.jpg", device="cpu")
    # 或通过 ExtractProcessor：
    ExtractProcessor.extract("img.jpg", device="cpu")
    # 首次运行会自动下载模型（~1.8GB），之后复用缓存。
    # 速度参考：M 系列芯片单张图约 5-15 秒。

Linux/Windows GPU（推荐生产部署）：
    1. 启动 vLLM 服务：
       paddleocr genai_server \\
           --model_name PaddleOCR-VL-1.5-0.9B \\
           --backend vllm --port 8118
    2. 初始化时传入：
       ImageExtractor("img.jpg",
                      vl_rec_backend="vllm-server",
                      vl_rec_server_url="http://localhost:8118/v1")

Linux GPU（native，无需额外服务，需 NVIDIA Compute Capability ≥ 7.0 + 12GB 显存）：
    直接调用，无需启动服务：
       ImageExtractor("img.jpg")  # vl_rec_backend="native" 为默认值
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

from .base import BaseExtractor
from models.document import Document

logger = logging.getLogger(__name__)

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

# ---------------------------------------------------------------------------
# 支持的格式
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 环境变量配置（优先级：代码传参 > 环境变量 > 默认值）
# ---------------------------------------------------------------------------
# 设置方式（任选其一）：
#   export VL_BACKEND=vllm-server
#   export VL_SERVER_URL=http://<Ubuntu公网IP>:8080/v1
#
# 或在项目根目录的 .env 文件里写（需要 python-dotenv）：
#   VL_BACKEND=vllm-server
#   VL_SERVER_URL=http://<Ubuntu公网IP>:8080/v1

def _env(key: str, default: str | None = None) -> str | None:
    """读环境变量，同时尝试从 .env 文件加载（如果安装了 python-dotenv）。"""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    return os.environ.get(key, default)


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
# 单例：PaddleOCRVL pipeline（进程内复用，避免重复加载模型权重）
# ---------------------------------------------------------------------------

from typing import TYPE_CHECKING, Any
if TYPE_CHECKING:
    from paddleocr import PaddleOCRVL as _PaddleOCRVLType

_pipeline: "Any | None" = None


def _get_pipeline(**init_kwargs) -> "Any":
    """
    返回全局 PaddleOCRVL 实例。
    首次调用时按 init_kwargs 初始化；后续调用忽略参数直接复用。
    如需更换后端，请重启进程或手动调用 reset_pipeline()。
    """
    global _pipeline
    if _pipeline is None:
        try:
            from paddleocr import PaddleOCRVL
        except ImportError as exc:
            raise ImportError(
                "PaddleOCR-VL 未安装，请执行: pip install \"paddleocr[doc-parser]\""
            ) from exc
        logger.info("[VL] 初始化 PaddleOCRVL，参数: %s", init_kwargs)
        _pipeline = PaddleOCRVL(**init_kwargs)
        logger.info("[VL] PaddleOCRVL 初始化完成，类型: %s", type(_pipeline).__name__)
    return _pipeline


def reset_pipeline() -> None:
    """强制下次调用时重新初始化 pipeline（用于切换后端或测试）。"""
    global _pipeline
    _pipeline = None
    logger.info("[VL] pipeline 已重置")


# ---------------------------------------------------------------------------
# 主类
# ---------------------------------------------------------------------------


class ImageExtractor(BaseExtractor):
    """
    基于 PaddleOCR-VL-1.5 的图片文字提取器。

    VLM 端到端处理，直接输出结构化文本，无需手工预处理。

    Parameters
    ----------
    file_path : str
        图片路径（本地文件或 HTTP URL）。
    max_file_mb : float
        本地文件大小上限（MB），默认 ``50``。URL 不受此限制。
    use_layout_detection : bool | None
        是否启用 PP-DocLayoutV3 布局检测。
        ``True`` = 强制开启（更准，稍慢）；
        ``None`` = 使用服务默认（推荐）。
    use_doc_orientation_classify : bool
        是否自动纠正文档方向（手机竖拍横向表格时有用），默认 ``False``。
    use_doc_unwarping : bool
        是否启用文档矫正展平（折叠/弯曲文档），默认 ``False``。
    vl_rec_backend : str | None
        VLM 推理后端。``None``（默认）= 不指定，由 PaddleOCRVL 根据环境自动选择。
        显式指定时支持：``"native"``（NVIDIA GPU）、``"vllm-server"``、
        ``"sglang-server"``、``"fastdeploy-server"``。
        **macOS 不需要指定此参数，配合 ``device="cpu"`` 即可。**
    vl_rec_server_url : str | None
        推理服务地址，使用外部服务后端时填写。
        vllm 示例：``"http://localhost:8118/v1"``
    max_pixels : int | None
        已废弃，保留仅为兼容旧调用，静默忽略。
    vl_rec_max_concurrency : int
        并发请求数，默认 ``1``（单图）。批量处理时可适当调高。
    device : str | None
        推理设备，例 ``"gpu:0"``、``"cpu"``。``None`` = 自动检测。
    output_format : str
        输出格式：``"markdown"``（默认）或 ``"text"``（纯文本，去除 Markdown 语法）。
    """

    def __init__(
        self,
        file_path: str,
        *,
        max_file_mb: float = 50.0,
        use_layout_detection: bool | None = None,
        use_doc_orientation_classify: bool = False,
        use_doc_unwarping: bool = False,
        vl_rec_backend: str | None = None,
        vl_rec_server_url: str | None = None,
        vl_rec_max_concurrency: int = 1,
        device: str | None = None,
        output_format: str = "markdown",
        max_pixels: int | None = None,   # 旧参数，静默忽略
        **_ignored,                       # 吸收其他未知旧参数
    ):
        self._file_path = file_path
        self._max_file_mb = max_file_mb
        self._use_layout_detection = use_layout_detection
        self._use_doc_orientation_classify = use_doc_orientation_classify
        self._use_doc_unwarping = use_doc_unwarping
        # 代码传参 > 环境变量 > None
        self._vl_rec_backend = vl_rec_backend or _env("VL_BACKEND")
        self._vl_rec_server_url = vl_rec_server_url or _env("VL_SERVER_URL")
        self._vl_rec_max_concurrency = vl_rec_max_concurrency
        self._device = device or _env("VL_DEVICE")
        self._output_format = output_format
        logger.debug(
            "[VL] 配置 backend=%s server_url=%s device=%s",
            self._vl_rec_backend, self._vl_rec_server_url, self._device,
        )

    # ------------------------------------------------------------------
    #  Public
    # ------------------------------------------------------------------

    def extract(self) -> list[Document]:
        path = Path(self._file_path)
        is_url = self._file_path.startswith(("http://", "https://"))
        ext = path.suffix.lower()

        # ── 本地文件：校验 + 预处理 ───────────────────────────────────
        infer_path: str = self._file_path  # 默认直接使用原文件
        tmp_path: Path | None = None       # 预处理生成的临时文件

        if not is_url:
            self._validate_image(path, ext)
            tmp_path = self._preprocess_image(path)
            if tmp_path is not None:
                infer_path = str(tmp_path)
                logger.info("[VL] 预处理完成，临时文件: %s", tmp_path.name)

        # ── 推理 ──────────────────────────────────────────────────────
        logger.info("[VL] 开始推理: %s", Path(infer_path).name)
        try:
            results = self._run_pipeline(infer_path)
        finally:
            # 无论成功失败都清理临时文件
            if tmp_path is not None and tmp_path.exists():
                try:
                    tmp_path.unlink()
                except Exception:
                    pass

        logger.info("[VL] 推理完成，共 %d 个结果块", len(results))

        # ── 拼装 Document ─────────────────────────────────────────────
        mime = _MIME_MAP.get(ext, "image/jpeg")
        docs = []
        for idx, (text, meta) in enumerate(results):
            logger.debug("[VL] 块 %d:\n%s", idx, text)
            docs.append(Document(
                page_content=text,
                metadata={
                    "source": self._file_path,
                    "mime_type": mime,
                    "block_index": idx,
                    **meta,
                },
            ))

        # 无内容时返回空 Document
        if not docs:
            logger.warning("[VL] 未识别到任何内容: %s", self._file_path)
            docs = [Document(
                page_content="",
                metadata={"source": self._file_path, "mime_type": mime},
            )]

        full_text = "\n\n".join(d.page_content for d in docs if d.page_content)
        logger.info("[VL] 最终输出 %d 字符", len(full_text))
        logger.debug("[VL] 完整输出:\n%s", full_text)

        return docs

    # ------------------------------------------------------------------
    #  本地文件校验（防御性，不做图像增强）                                #
    # ------------------------------------------------------------------

    # magic bytes → 格式映射（不依赖扩展名）
    _MAGIC: list[tuple[bytes, str]] = [
        (b"\xff\xd8\xff",          "image/jpeg"),
        (b"\x89PNG\r\n\x1a\n",   "image/png"),
        (b"BM",                        "image/bmp"),
        (b"RIFF",                      "image/webp"),   # RIFF????WEBP
        (b"II\x2a\x00",             "image/tiff"),   # little-endian TIFF
        (b"MM\x00\x2a",             "image/tiff"),   # big-endian TIFF
        (b"GIF87a",                    "image/gif"),
        (b"GIF89a",                    "image/gif"),
    ]

    def _validate_image(self, path: Path, ext: str) -> None:
        """
        本地文件防御性校验，仅报错，不修改图片内容。

        校验顺序：
          1. 文件存在且非空
          2. 扩展名在支持列表内
          3. 文件大小不超过 max_file_mb
          4. magic bytes 与扩展名匹配（防止把 PDF/文档误传进来）
          5. cv2 可解码（图片未损坏）
        """
        # 1. 文件存在 & 非空
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {path}")
        file_mb = path.stat().st_size / (1024 * 1024)
        if file_mb == 0:
            raise ValueError(f"文件为空: {path}")

        # 2. 扩展名校验
        if ext not in _MIME_MAP:
            raise ValueError(
                f"不支持的扩展名 {ext!r}，支持: {', '.join(sorted(_MIME_MAP))}"
            )

        # 3. 文件大小
        logger.info("[VL] 文件: %s，大小: %.1f MB", path.name, file_mb)
        if file_mb > self._max_file_mb:
            raise ValueError(
                f"文件过大（{file_mb:.1f} MB > {self._max_file_mb} MB）: {path}"
            )

        # 4. magic bytes 校验（防止扩展名伪装）
        header = path.read_bytes()[:12]
        detected_mime: str | None = None
        for magic, mime in self._MAGIC:
            if header.startswith(magic):
                detected_mime = mime
                break
        # WEBP 需要额外确认第 8-11 字节
        if detected_mime == "image/webp" and header[8:12] != b"WEBP":
            detected_mime = None

        if detected_mime is None:
            raise ValueError(
                f"文件内容不是有效图片（magic bytes 未匹配），"
                f"请确认文件未损坏: {path.name}"
            )
        expected_mime = _MIME_MAP[ext]
        # GIF/TIFF 允许宽松匹配（扩展名 .tif/.tiff 都指向同一 mime）
        if detected_mime != expected_mime:
            logger.warning(
                "[VL] 扩展名 %s 与文件实际格式不符（检测到 %s），尝试继续处理",
                ext, detected_mime,
            )

        # 5. cv2 可解码性校验（捕获损坏文件）
        try:
            import cv2
            import numpy as np
            raw = np.frombuffer(path.read_bytes(), dtype=np.uint8)
            probe = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
            if probe is None:
                raise ValueError(f"图片文件损坏或无法解码: {path.name}")
            h, w = probe.shape[:2]
            logger.info("[VL] 图片尺寸: %dx%d px", w, h)
            del probe, raw
        except ImportError:
            # cv2 未安装时跳过解码校验，不阻断流程
            logger.debug("[VL] opencv 未安装，跳过解码校验")


    # ------------------------------------------------------------------
    #  图片预处理（通用，对所有图片无损等价）                               #
    # ------------------------------------------------------------------

    def _preprocess_image(self, path: Path) -> Path | None:
        """
        通用图片预处理，返回处理后的临时文件路径。
        若图片无需任何处理则返回 None（直接用原文件）。

        处理内容（按需，不强制）：
          1. EXIF 方向修正：手机拍摄的图大多带旋转标记，VLM 看到翻转图
             识别率会下降。Pillow 的 ImageOps.exif_transpose 一行解决。
          2. 超大图降采样：VLM 支持动态分辨率，但原图过大（>4000px 长边）
             会显著拖慢推理。用 INTER_AREA 等比缩到 OCR_MAX_SIDE 以内。
             OCR_MAX_SIDE 默认 3500，低于 PaddleOCR 内部的 4000 警告线。

        不做的事：
          - 不做对比度/亮度增强（VLM 自身鲁棒性足够）
          - 不做灰度转换（VLM 需要彩色信息辅助理解版式）
          - 不做裁剪/旋转业务逻辑（与内容无关的预处理才放这里）
        """
        import tempfile
        import uuid

        try:
            from PIL import Image, ImageOps
        except ImportError:
            logger.debug("[VL] Pillow 未安装，跳过预处理")
            return None

        img = Image.open(path)
        original_size = img.size  # (w, h)
        changed = False

        # ── 1. EXIF 方向修正 ──────────────────────────────────────────
        try:
            img_oriented = ImageOps.exif_transpose(img)
            if img_oriented is not img:
                img = img_oriented
                changed = True
                logger.info(
                    "[VL] EXIF 方向已修正: %s %s → %s",
                    path.name, original_size, img.size,
                )
        except Exception as e:
            logger.debug("[VL] EXIF 修正跳过: %s", e)

        # ── 2. 超大图降采样 ───────────────────────────────────────────
        ocr_max_side = 3500
        w, h = img.size
        if max(w, h) > ocr_max_side:
            scale = ocr_max_side / max(w, h)
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            img = img.resize((new_w, new_h), Image.LANCZOS)
            changed = True
            logger.info(
                "[VL] 超大图降采样: %s %dx%d → %dx%d",
                path.name, w, h, new_w, new_h,
            )

        if not changed:
            img.close()
            return None

        # ── 写临时文件（保留原格式，JPEG 用 quality=92）─────────────
        suffix = path.suffix.lower() or ".jpg"
        tmp = Path(tempfile.gettempdir()) / f"_vlocr_{uuid.uuid4().hex}{suffix}"
        save_kwargs: dict = {}
        if suffix in (".jpg", ".jpeg"):
            save_kwargs["quality"] = 92
            save_kwargs["optimize"] = True
        img.save(tmp, **save_kwargs)
        img.close()
        logger.debug("[VL] 预处理临时文件: %s (%.1f MB)", tmp.name,
                     tmp.stat().st_size / 1024 / 1024)
        return tmp

    # ------------------------------------------------------------------
    #  Pipeline 调用
    # ------------------------------------------------------------------

    def _build_init_kwargs(self) -> dict:
        """构建 PaddleOCRVL 初始化参数。"""
        kwargs: dict = {
            "vl_rec_max_concurrency": self._vl_rec_max_concurrency,
            "use_doc_orientation_classify": self._use_doc_orientation_classify,
            "use_doc_unwarping": self._use_doc_unwarping,
        }
        # vl_rec_backend 为 None 时不传，让 PaddleOCRVL 自动选择
        # macOS 不传此参数 + device="cpu" 即可正常运行
        if self._vl_rec_backend is not None:
            kwargs["vl_rec_backend"] = self._vl_rec_backend
        if self._vl_rec_server_url:
            kwargs["vl_rec_server_url"] = self._vl_rec_server_url
        if self._device:
            kwargs["device"] = self._device
        if self._use_layout_detection is not None:
            kwargs["use_layout_detection"] = self._use_layout_detection
        return kwargs

    def _run_pipeline(self, infer_path: str) -> list[tuple[str, dict]]:
        """
        调用 PaddleOCRVL pipeline，返回 [(text, metadata), ...] 列表。

        每个元素对应一个布局块（text / table / formula / chart / seal 等）。
        metadata 包含 label（块类型）和可选的置信度等信息。
        infer_path 是经过预处理后实际送入推理的文件路径（可能是临时文件）。
        """
        pipeline = _get_pipeline(**self._build_init_kwargs())

        # 防御：确认 pipeline 有 predict 方法
        if not hasattr(pipeline, "predict"):
            raise RuntimeError(
                f"PaddleOCRVL 实例（{type(pipeline).__name__}）没有 predict 方法，"
                f"请确认 paddleocr 版本 >= 3.0: pip install -U paddleocr"
            )

        logger.info("[VL] 调用 pipeline.predict，文件: %s", infer_path)
        try:
            raw_results = list(pipeline.predict(infer_path))  # type: ignore[union-attr]
        except Exception as exc:
            logger.error("[VL] pipeline.predict 失败: %s", exc, exc_info=True)
            raise

        logger.info("[VL] predict 返回 %d 个结果", len(raw_results))
        return self._parse_results(raw_results)

    # ------------------------------------------------------------------
    #  结果解析
    # ------------------------------------------------------------------

    @staticmethod
    def _diagnose_result(result) -> None:
        """DEBUG 专用：打印 result 对象的完整结构，帮助定位字段名。"""
        import json, tempfile, os
        logger.info("[VL-DIAG] result 类型: %s", type(result).__name__)
        logger.info("[VL-DIAG] dir(result): %s",
                    [a for a in dir(result) if not a.startswith("__")])

        # 打印 .res 属性
        if hasattr(result, "res"):
            res = result.res
            logger.info("[VL-DIAG] result.res 类型: %s", type(res).__name__)
            if isinstance(res, dict):
                logger.info("[VL-DIAG] result.res keys: %s", list(res.keys()))
                # 每个 key 的前 200 字符
                for k, v in res.items():
                    preview = str(v)[:200]
                    logger.info("[VL-DIAG]   res[%r] = %s", k, preview)

        # 尝试 save_to_json 写临时文件
        for method in ("save_to_json", "save_to_markdown", "save_to_img"):
            if hasattr(result, method):
                logger.info("[VL-DIAG] result 有方法: %s()", method)
                if method == "save_to_json":
                    try:
                        tmp = tempfile.mktemp(suffix=".json")
                        getattr(result, method)(tmp)
                        with open(tmp) as f:
                            data = json.load(f)
                        logger.info("[VL-DIAG] save_to_json 内容 keys: %s", list(data.keys()) if isinstance(data, dict) else type(data).__name__)
                        logger.info("[VL-DIAG] save_to_json 内容前500字符: %s", json.dumps(data, ensure_ascii=False)[:500])
                        os.unlink(tmp)
                    except Exception as e:
                        logger.info("[VL-DIAG] save_to_json 失败: %s", e)

    def _parse_results(self, raw_results: list) -> list[tuple[str, dict]]:
        """
        将 PaddleOCRVL 返回的 result 列表解析为 (text, metadata) 元组列表。
        优先从 result.res 中提取结构化内容；兜底用 save_to_json 写临时文件读取。
        """
        output: list[tuple[str, dict]] = []

        for result in raw_results:
            logger.info("[VL] result 类型: %s", type(result).__name__)

            # 仅在 DEBUG 级别时打印完整结构（生产环境不影响性能）
            if logger.isEnabledFor(logging.DEBUG):
                self._diagnose_result(result)

            # ── 方式 1：从 result.res 按块提取 ──────────────────────
            blocks = self._extract_blocks(result)
            if blocks:
                output.extend(blocks)
                continue

            # ── 方式 2：save_to_json 写临时文件兜底 ─────────────────
            md = self._result_to_markdown(result)
            if md:
                text = md if self._output_format == "markdown" else _strip_markdown(md)
                output.append((text.strip(), {"label": "page"}))

        return output

    def _extract_blocks(self, result) -> list[tuple[str, dict]]:
        """
        从 PaddleOCRVLResult 中提取文本块。

        result 是一个 dict-like 对象，关键字段：
          - parsing_res_list: 主要内容列表，每项有 label / bbox / content
          - layout_det_res:   版面检测结果（boxes）
          - table_res_list:   表格识别结果
        """
        blocks_out: list[tuple[str, dict]] = []

        try:
            # result 本身就是 dict（PaddleOCRVLResult 实现了 __getitem__）
            res_dict = dict(result) if hasattr(result, "keys") else {}
            if not res_dict and hasattr(result, "res"):
                res_dict = result.res or {}

            # ── 优先读 parsing_res_list（主要识别结果）─────────────
            parsing_res_list = res_dict.get("parsing_res_list", [])
            if parsing_res_list:
                logger.info("[VL] 解析 parsing_res_list，共 %d 块", len(parsing_res_list))
                for blk in parsing_res_list:
                    # blk 可能是 dict 也可能是有属性的对象
                    if isinstance(blk, dict):
                        label = blk.get("label", "text")
                        text = (
                            blk.get("content")
                            or blk.get("markdown")
                            or blk.get("text")
                            or ""
                        ).strip()
                        bbox = blk.get("bbox", [])
                    else:
                        label = getattr(blk, "label", "text")
                        text = (
                            getattr(blk, "content", "")
                            or getattr(blk, "markdown", "")
                            or getattr(blk, "text", "")
                            or ""
                        ).strip()
                        bbox = getattr(blk, "bbox", [])

                    if not text:
                        continue

                    if self._output_format == "text":
                        text = _strip_markdown(text)

                    meta = {"label": label}
                    if bbox:
                        meta["bbox"] = bbox

                    logger.debug("[VL] 块 label=%s 内容前80字: %r", label, text[:80])
                    blocks_out.append((text, meta))

                if blocks_out:
                    return blocks_out

            # ── 兜底读 layout_det_res 的 boxes（只有坐标没有文本，跳过）
            # ── 再兜底读旧版 blocks 字段 ─────────────────────────────
            for key in ("blocks", "ocr_res", "rec_res"):
                items = res_dict.get(key, [])
                if not items:
                    continue
                logger.info("[VL] 从 %r 字段读取 %d 块", key, len(items))
                for blk in items:
                    if not isinstance(blk, dict):
                        continue
                    label = blk.get("label", "text")
                    text = (
                        blk.get("content")
                        or blk.get("markdown")
                        or blk.get("text")
                        or ""
                    ).strip()
                    if not text:
                        continue
                    if self._output_format == "text":
                        text = _strip_markdown(text)
                    blocks_out.append((text, {"label": label}))
                if blocks_out:
                    return blocks_out

        except Exception as exc:
            logger.warning("[VL] 块解析失败，将用整页兜底: %s", exc)
            return []

        return blocks_out

    def _result_to_markdown(self, result) -> str:
        """整页兜底：save_to_json 写临时文件 → 读取 → 拼文本。"""
        import json, tempfile, os

        # 方式 A：result.res 直接含文本字段
        res_dict = result.res if hasattr(result, "res") else {}
        if isinstance(res_dict, dict):
            for key in ("markdown", "text", "content", "rec_text", "ocr_text"):
                val = res_dict.get(key, "")
                if val and isinstance(val, str) and val.strip():
                    logger.info("[VL] 从 result.res[%r] 取到文本，%d 字符", key, len(val))
                    return val

        # 方式 B：save_to_json 写临时文件
        if hasattr(result, "save_to_json"):
            tmp = None
            try:
                tmp = tempfile.mktemp(suffix=".json")
                result.save_to_json(tmp)
                with open(tmp, encoding="utf-8") as f:
                    data = json.load(f)
                logger.info("[VL] save_to_json keys: %s", list(data.keys()) if isinstance(data, dict) else type(data))
                texts = []
                if isinstance(data, dict):
                    # 尝试常见顶层文本字段
                    for key in ("markdown", "text", "content"):
                        val = data.get(key, "")
                        if val and isinstance(val, str) and val.strip():
                            return val
                    # 尝试 blocks 数组
                    for blk in data.get("blocks", []):
                        t = blk.get("markdown") or blk.get("text") or blk.get("content") or ""
                        if t.strip():
                            texts.append(t.strip())
                return "\n\n".join(texts)
            except Exception as exc:
                logger.warning("[VL] save_to_json 兜底失败: %s", exc)
            finally:
                if tmp and os.path.exists(tmp):
                    try: os.unlink(tmp)
                    except Exception: pass

        # 方式 C：save_to_markdown 写临时文件
        if hasattr(result, "save_to_markdown"):
            tmp = None
            try:
                tmp = tempfile.mktemp(suffix=".md")
                result.save_to_markdown(tmp)
                with open(tmp, encoding="utf-8") as f:
                    return f.read()
            except Exception as exc:
                logger.warning("[VL] save_to_markdown 兜底失败: %s", exc)
            finally:
                if tmp and os.path.exists(tmp):
                    try: os.unlink(tmp)
                    except Exception: pass

        return ""


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------


def _strip_markdown(text: str) -> str:
    """
    去除常见 Markdown 语法，输出纯文本。
    仅做简单清理，不依赖外部库。
    """
    # 去掉代码块 ``` ```
    text = re.sub(r"```[^\n]*\n(.*?)```", r"\1", text, flags=re.DOTALL)
    # 去掉标题 # ## ###
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # 去掉加粗/斜体
    text = re.sub(r"\*{1,3}(.*?)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,3}(.*?)_{1,3}", r"\1", text)
    # 去掉行内代码
    text = re.sub(r"`([^`]+)`", r"\1", text)
    # 表格分隔行
    text = re.sub(r"^\|?[-:| ]+\|?\s*$", "", text, flags=re.MULTILINE)
    # 压缩多余空行
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
    