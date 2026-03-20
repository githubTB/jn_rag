"""
tests/test_pipeline.py — 完整流程测试。

覆盖范围：保存文件到磁盘之后的全链路
    解析 → 切片 → Chunk 级去重 → 向量化

运行方式：
    # 项目根目录下
    python tests/test_pipeline.py
    python tests/test_pipeline.py uploads/2020年水电气汇总统计表.jpg
    python tests/test_pipeline.py uploads/澳龙生物营业执照（副本）.jpg --dry-run
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# ── sys.path 必须在所有项目 import 之前 ──────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

# ── 日志配置（DEBUG 级别，完整链路可见）──────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)5s] %(name)s:%(lineno)d - %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)

# 第三方库的 DEBUG 日志太啰嗦，只看项目自己的
for noisy in ("PIL", "urllib3", "httpx", "httpcore",
              "paddleocr", "paddlex", "paddle"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

logger = logging.getLogger("test_pipeline")

# ── 项目 import ───────────────────────────────────────────────────────────
from config.settings import settings
from extract_processor import ExtractProcessor
from extractor.doc_chunker import DocChunker
from core.dedup import Dedup
from core.embedder import Embedder


# ---------------------------------------------------------------------------
# 测试入口
# ---------------------------------------------------------------------------

def run_pipeline(file_path: str, dry_run: bool = False) -> None:
    """
    从磁盘文件开始跑完整链路。

    dry_run=True 时跳过向量化（不需要 BGE-M3 模型，方便快速验证前几步）。
    """
    path = Path(file_path)
    sep = "=" * 60

    logger.info("%s", sep)
    logger.info("测试文件: %s", path.name)
    logger.info("文件大小: %.2f MB", path.stat().st_size / 1024 / 1024)
    logger.info("dry_run : %s", dry_run)
    logger.info("%s", sep)

    total_start = time.perf_counter()

    # ── STEP 1: 解析 ────────────────────────────────────────────────────
    logger.info("")
    logger.info("── STEP 1: 解析 ──────────────────────────────────")
    t0 = time.perf_counter()

    docs = ExtractProcessor.extract(
        str(path),
        # 图片 OCR 参数（非图片文件会被自动忽略）
        vl_rec_backend=settings.vl_backend,
        vl_rec_server_url=settings.vl_server_url,
        device=settings.vl_device,
        max_file_mb=settings.vl_max_file_mb,
    )

    elapsed = time.perf_counter() - t0
    logger.info("解析完成: %d 个 Document，耗时 %.2fs", len(docs), elapsed)

    for i, doc in enumerate(docs):
        logger.debug(
            "  doc[%d] label=%s  字符数=%d  内容前100字: %r",
            i,
            doc.metadata.get("label", "unknown"),
            len(doc.page_content),
            doc.page_content[:100],
        )

    assert docs, "解析结果不能为空"

    # ── STEP 2: 切片 ────────────────────────────────────────────────────
    logger.info("")
    logger.info("── STEP 2: 切片 ──────────────────────────────────")
    t0 = time.perf_counter()

    chunks = DocChunker.chunk(docs, source_override=str(path))

    elapsed = time.perf_counter() - t0
    logger.info("切片完成: %d 个 chunk，耗时 %.2fs", len(chunks), elapsed)

    for i, chunk in enumerate(chunks):
        logger.debug(
            "  chunk[%d] strategy=%s  字符数=%d  内容前80字: %r",
            i,
            chunk.metadata.get("chunk_strategy", "unknown"),
            len(chunk.page_content),
            chunk.page_content[:80],
        )

    assert chunks, "切片结果不能为空"

    # ── STEP 3: Chunk 级去重 ────────────────────────────────────────────
    logger.info("")
    logger.info("── STEP 3: Chunk 级去重 ──────────────────────────")
    t0 = time.perf_counter()

    # 用文件 hash 作为 file_id（不需要先 register_file，只做 chunk 级去重测试）
    file_id = Dedup.hash_file(path)
    logger.debug("file_id (SHA256前12位): %s", file_id[:12])

    new_chunks, new_hashes = Dedup.filter_new_chunks(chunks, file_id)

    elapsed = time.perf_counter() - t0
    logger.info(
        "去重完成: 输入 %d 个，跳过 %d 个，新增 %d 个，耗时 %.2fs",
        len(chunks),
        len(chunks) - len(new_chunks),
        len(new_chunks),
        elapsed,
    )

    for i, (chunk, h) in enumerate(zip(new_chunks, new_hashes)):
        logger.debug(
            "  new_chunk[%d] hash=%s  内容前60字: %r",
            i, h[:12], chunk.page_content[:60],
        )

    # ── STEP 4: 向量化 ──────────────────────────────────────────────────
    logger.info("")
    logger.info("── STEP 4: 向量化 ────────────────────────────────")

    if dry_run:
        logger.info("dry_run=True，跳过向量化（不加载 BGE-M3 模型）")
        vectors = []
    elif not new_chunks:
        logger.info("无新 chunk，跳过向量化")
        vectors = []
    else:
        t0 = time.perf_counter()
        vectors = Embedder.embed(new_chunks)
        elapsed = time.perf_counter() - t0
        logger.info(
            "向量化完成: %d 个向量，维度=%d，耗时 %.2fs",
            len(vectors),
            len(vectors[0]) if vectors else 0,
            elapsed,
        )

        for i, vec in enumerate(vectors):
            logger.debug(
                "  vector[%d] 前5维: %s ...",
                i, [round(v, 4) for v in vec[:5]],
            )

        assert len(vectors) == len(new_chunks), "向量数量与 chunk 数量不一致"

    # ── 汇总 ────────────────────────────────────────────────────────────
    logger.info("")
    logger.info("%s", sep)
    total_elapsed = time.perf_counter() - total_start
    logger.info("全链路完成，总耗时 %.2fs", total_elapsed)
    logger.info("  解析: %d docs", len(docs))
    logger.info("  切片: %d chunks", len(chunks))
    logger.info("  去重后新增: %d chunks", len(new_chunks))
    logger.info("  向量化: %d vectors", len(vectors))
    logger.info("%s", sep)

    # ── 第二次运行验证去重生效 ───────────────────────────────────────────
    if new_chunks and not dry_run:
        logger.info("")
        logger.info("── 二次去重验证（模拟重复上传）──────────────────")

        # 先登记这批 chunk
        Dedup.register_chunks([
            (h, file_id, c.metadata.get("chunk_index", i))
            for i, (c, h) in enumerate(zip(new_chunks, new_hashes))
        ])

        # 再跑一次 filter，应该全部被过滤掉
        dup_chunks, dup_hashes = Dedup.filter_new_chunks(chunks, file_id)
        logger.info("二次去重结果: 新增 %d 个（预期 0）", len(dup_chunks))
        assert len(dup_chunks) == 0, f"二次去重应过滤全部，但还剩 {len(dup_chunks)} 个"
        logger.info("✓ 去重验证通过")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="完整 RAG 流程测试")
    parser.add_argument(
        "file",
        nargs="?",
        default=None,
        help="测试文件路径（默认自动查找 uploads/ 下第一个支持的文件）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="跳过向量化步骤（不加载 BGE-M3，快速验证解析/切片/去重）",
    )
    args = parser.parse_args()

    # 自动查找测试文件
    if args.file:
        file_path = args.file
    else:
        upload_dir = Path(settings.upload_dir)
        supported = {".jpg", ".jpeg", ".png", ".pdf", ".docx",
                     ".xlsx", ".csv", ".txt", ".md"}
        candidates = [
            f for f in upload_dir.iterdir()
            if f.is_file() and f.suffix.lower() in supported
        ]
        if not candidates:
            logger.error("uploads/ 目录下没有找到支持的测试文件")
            logger.error("请指定文件路径: python tests/test_pipeline.py <file>")
            sys.exit(1)
        # 优先选图片
        images = [f for f in candidates
                  if f.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        file_path = str(images[0] if images else candidates[0])
        logger.info("自动选择测试文件: %s", file_path)

    if not Path(file_path).exists():
        logger.error("文件不存在: %s", file_path)
        sys.exit(1)

    try:
        run_pipeline(file_path, dry_run=args.dry_run)
        logger.info("✓ 所有步骤通过")
    except AssertionError as e:
        logger.error("✗ 断言失败: %s", e)
        sys.exit(1)
    except Exception as e:
        logger.exception("✗ 异常: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
