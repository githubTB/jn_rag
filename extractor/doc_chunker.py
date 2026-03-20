"""
DocChunker — 文档切片模块。

根据 PaddleOCR-VL 返回的块类型（label）选择不同切片策略，
输出适合 RAG 向量化的 Document 列表。

切片策略
--------
- whole       : 整块作为一个 chunk（表格、铭牌、营业执照等）
- by_paragraph: 按段落切（普通文本、发票备注等）
- skip        : 跳过（图片块、页眉页脚等无意义内容）

使用示例
--------
    from extract_processor import ExtractProcessor
    from extractor.doc_chunker import DocChunker

    raw_docs = ExtractProcessor.extract("invoice.jpg")
    chunks = DocChunker.chunk(raw_docs)
    # chunks 即可直接送入向量数据库
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from models.document import Document

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 块类型 → 切片策略映射
# ---------------------------------------------------------------------------
# PaddleOCR-VL 返回的 label 值：
#   ocr / text / table / figure / title / formula / seal / chart 等
#
# whole        : 整块不切，保留完整上下文（推荐用于结构化文档）
# by_paragraph : 按空行/换行切段（适合长文本）
# skip         : 忽略此块

CHUNK_STRATEGY: dict[str, str] = {
    "ocr":       "whole",         # 表格/铭牌/营业执照 → 整块保留
    "table":     "whole",         # 识别为表格的块
    "text":      "by_paragraph",  # 普通文本段落
    "title":     "whole",         # 标题单独一块
    "formula":   "whole",         # 公式
    "seal":      "whole",         # 印章
    "chart":     "skip",          # 图表（纯图，无文字价值）
    "figure":    "skip",          # 插图
    "header":    "skip",          # 页眉
    "footer":    "skip",          # 页脚
    "number":    "skip",          # 页码
    "footnote":  "by_paragraph",  # 脚注
    "page":      "by_paragraph",  # 整页兜底（无布局检测时）
    # 未知类型默认 whole
    "_default":  "whole",
}

# ---------------------------------------------------------------------------
# 切片参数
# ---------------------------------------------------------------------------
# by_paragraph 模式下，单个 chunk 的最大字符数
MAX_PARAGRAPH_CHARS = 500

# whole 模式下，超过此长度才考虑按段落二次切分
WHOLE_MAX_CHARS = 2000


class DocChunker:
    """
    文档切片器。

    根据 CHUNK_STRATEGY 对 PaddleOCR-VL 输出的 Document 列表做切片，
    每个输出 chunk 带完整 metadata，可直接送入向量数据库。
    """

    @classmethod
    def chunk(
        cls,
        docs: list[Document],
        source_override: str | None = None,
    ) -> list[Document]:
        """
        对 ExtractProcessor.extract() 的输出做切片。

        Parameters
        ----------
        docs : list[Document]
            ExtractProcessor.extract() 返回的原始 Document 列表。
        source_override : str | None
            覆盖 metadata 里的 source 字段（可选）。

        Returns
        -------
        list[Document]
            切片后的 Document 列表，每个 chunk 含完整 metadata。
        """
        chunks: list[Document] = []

        for doc in docs:
            label = doc.metadata.get("label", "_default")
            strategy = CHUNK_STRATEGY.get(label, CHUNK_STRATEGY["_default"])
            source = source_override or doc.metadata.get("source", "")
            text = doc.page_content.strip()

            if not text:
                continue

            logger.debug(
                "[Chunker] label=%s strategy=%s 字符数=%d source=%s",
                label, strategy, len(text), Path(source).name,
            )

            if strategy == "skip":
                logger.debug("[Chunker] 跳过块 label=%s", label)
                continue

            elif strategy == "whole":
                # 整块作为一个 chunk
                # 超长时二次按段落切（避免单个 chunk 超出 embedding 模型限制）
                if len(text) > WHOLE_MAX_CHARS:
                    sub_chunks = cls._split_by_paragraph(text, MAX_PARAGRAPH_CHARS)
                    for i, sub in enumerate(sub_chunks):
                        chunks.append(Document(
                            page_content=sub,
                            metadata={
                                **doc.metadata,
                                "source": source,
                                "chunk_index": i,
                                "chunk_total": len(sub_chunks),
                                "chunk_strategy": "whole_split",
                            },
                        ))
                else:
                    chunks.append(Document(
                        page_content=text,
                        metadata={
                            **doc.metadata,
                            "source": source,
                            "chunk_index": 0,
                            "chunk_total": 1,
                            "chunk_strategy": "whole",
                        },
                    ))

            elif strategy == "by_paragraph":
                paragraphs = cls._split_by_paragraph(text, MAX_PARAGRAPH_CHARS)
                for i, para in enumerate(paragraphs):
                    chunks.append(Document(
                        page_content=para,
                        metadata={
                            **doc.metadata,
                            "source": source,
                            "chunk_index": i,
                            "chunk_total": len(paragraphs),
                            "chunk_strategy": "by_paragraph",
                        },
                    ))

        logger.info(
            "[Chunker] 输入 %d 块 → 输出 %d 个 chunk",
            len(docs), len(chunks),
        )
        return chunks

    @staticmethod
    def _split_by_paragraph(text: str, max_chars: int) -> list[str]:
        """
        按空行或换行切段，超过 max_chars 的段再按句子切分。
        """
        # 先按空行切
        raw_paras = re.split(r"\n{2,}", text.strip())
        result: list[str] = []
        buffer = ""

        for para in raw_paras:
            para = para.strip()
            if not para:
                continue

            # 段落本身不超长，直接累积
            if len(buffer) + len(para) + 1 <= max_chars:
                buffer = (buffer + "\n" + para).strip()
            else:
                if buffer:
                    result.append(buffer)
                # 段落本身超长，按句子切
                if len(para) > max_chars:
                    sentences = re.split(r"(?<=[。！？.!?])", para)
                    sent_buf = ""
                    for sent in sentences:
                        if len(sent_buf) + len(sent) <= max_chars:
                            sent_buf += sent
                        else:
                            if sent_buf:
                                result.append(sent_buf.strip())
                            sent_buf = sent
                    if sent_buf:
                        buffer = sent_buf.strip()
                    else:
                        buffer = ""
                else:
                    buffer = para

        if buffer:
            result.append(buffer)

        return [r for r in result if r.strip()]
        