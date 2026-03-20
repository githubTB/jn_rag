"""
core/dedup.py — 去重工具类。

与项目其他工具类风格一致（类方法静态调用，无需实例化）：
    ExtractProcessor.extract()      → 解析
    DocChunker.chunk()              → 切片
    Embedder.embed()                → 向量化
    Dedup.is_file_new()             → 去重检查  ← 本模块

数据库由模块级单例管理，进程内只初始化一次。

两级去重
--------
1. 文件级：对文件内容计算 SHA256，同一内容不重复处理（不管文件名）
2. Chunk 级：对 chunk 文本计算 SHA256，同一内容不重复入向量库

使用示例
--------
    from core.dedup import Dedup

    file_path = "uploads/invoice.jpg"

    # 文件级去重（已处理完成则跳过）
    if not Dedup.is_file_new(file_path):
        return

    # 登记文件（状态 pending）
    file_id = Dedup.register_file(file_path)

    # 解析 → 切片
    docs   = ExtractProcessor.extract(file_path)
    chunks = DocChunker.chunk(docs)

    # Chunk 级去重
    new_chunks, new_hashes = Dedup.filter_new_chunks(chunks, file_id)

    # ... 向量化 → 入库 ...

    # 完成后标记
    Dedup.mark_done(file_id)
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from config.settings import settings
from models.document import Document

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 模块级数据库单例
# ---------------------------------------------------------------------------

_initialized: bool = False


def _get_db_path() -> str:
    return settings.db_path


def _ensure_init() -> None:
    """建表（幂等）。"""
    global _initialized
    if _initialized:
        return
    with _conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS files (
                id          TEXT PRIMARY KEY,
                filename    TEXT NOT NULL,
                file_path   TEXT NOT NULL,
                status      TEXT NOT NULL DEFAULT 'pending',
                created_at  TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chunks (
                id          TEXT PRIMARY KEY,
                file_id     TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                created_at  TEXT NOT NULL,
                FOREIGN KEY (file_id) REFERENCES files(id)
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_file_id ON chunks(file_id);
        """)
    _initialized = True
    logger.debug("[Dedup] 初始化完成: %s", _get_db_path())


@contextmanager
def _conn():
    conn = sqlite3.connect(_get_db_path(), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 工具类
# ---------------------------------------------------------------------------


class Dedup:
    """
    去重工具类，所有方法均为类方法，无需实例化。

    完整链路：
        file_id = Dedup.register_file(path)
        new_chunks, hashes = Dedup.filter_new_chunks(chunks, file_id)
        # ... 向量化入库 ...
        Dedup.mark_done(file_id)
    """

    # ------------------------------------------------------------------
    #  Hash
    # ------------------------------------------------------------------

    @staticmethod
    def hash_file(file_path: str | Path) -> str:
        """文件内容 SHA256（分块读取，大文件安全）。"""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    @staticmethod
    def hash_text(text: str) -> str:
        """文本内容 SHA256。"""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    #  文件级
    # ------------------------------------------------------------------

    @classmethod
    def is_file_new(cls, file_path: str | Path) -> bool:
        """
        True  → 未入库或上次失败，需要处理。
        False → 已成功完成，可跳过。
        """
        _ensure_init()
        file_id = cls.hash_file(file_path)
        with _conn() as conn:
            row = conn.execute(
                "SELECT status FROM files WHERE id = ?", (file_id,)
            ).fetchone()
        if row is None:
            return True
        return row["status"] != "done"

    @classmethod
    def register_file(
        cls,
        file_path: str | Path,
        filename: str | None = None,
    ) -> str:
        """
        登记文件，返回 file_id（SHA256）。
        已存在时重置为 pending（支持重试）。
        """
        _ensure_init()
        path = Path(file_path)
        file_id = cls.hash_file(path)
        fname = filename or path.name
        now = datetime.now(timezone.utc).isoformat()

        with _conn() as conn:
            conn.execute(
                """
                INSERT INTO files (id, filename, file_path, status, created_at)
                VALUES (?, ?, ?, 'pending', ?)
                ON CONFLICT(id) DO UPDATE SET
                    status    = 'pending',
                    file_path = excluded.file_path,
                    filename  = excluded.filename
                """,
                (file_id, fname, str(path), now),
            )
        logger.info("[Dedup] 登记文件: %s  id=%s", fname, file_id[:12])
        return file_id

    @classmethod
    def mark_done(cls, file_id: str) -> None:
        """标记文件处理成功。"""
        _ensure_init()
        with _conn() as conn:
            conn.execute(
                "UPDATE files SET status = 'done' WHERE id = ?", (file_id,)
            )
        logger.info("[Dedup] 文件完成: id=%s", file_id[:12])

    @classmethod
    def mark_failed(cls, file_id: str) -> None:
        """标记处理失败（下次 is_file_new() 返回 True，允许重试）。"""
        _ensure_init()
        with _conn() as conn:
            conn.execute(
                "UPDATE files SET status = 'failed' WHERE id = ?", (file_id,)
            )
        logger.warning("[Dedup] 文件失败: id=%s", file_id[:12])

    # ------------------------------------------------------------------
    #  Chunk 级
    # ------------------------------------------------------------------

    @classmethod
    def filter_new_chunks(
        cls,
        chunks: list[Document],
        file_id: str,
    ) -> tuple[list[Document], list[str]]:
        """
        过滤已入库 chunk，返回 (新chunk列表, 对应hash列表)，顺序一致。

        典型用法：
            new_chunks, hashes = Dedup.filter_new_chunks(chunks, file_id)
            vectors = Embedder.embed(new_chunks)
            for chunk, vec, h in zip(new_chunks, vectors, hashes):
                # 写 Milvus...
            Dedup.register_chunks(
                [(h, file_id, c.metadata["chunk_index"])
                 for c, h in zip(new_chunks, hashes)]
            )
        """
        _ensure_init()
        new_chunks: list[Document] = []
        new_hashes: list[str] = []

        for chunk in chunks:
            h = cls.hash_text(chunk.page_content)
            with _conn() as conn:
                exists = conn.execute(
                    "SELECT 1 FROM chunks WHERE id = ?", (h,)
                ).fetchone()
            if not exists:
                new_chunks.append(chunk)
                new_hashes.append(h)
            else:
                logger.debug("[Dedup] chunk 已存在，跳过: %s", h[:12])

        logger.info(
            "[Dedup] chunk 去重: 共 %d，跳过 %d，新增 %d",
            len(chunks), len(chunks) - len(new_chunks), len(new_chunks),
        )
        return new_chunks, new_hashes

    @classmethod
    def register_chunks(
        cls,
        chunk_records: list[tuple[str, str, int]],
    ) -> None:
        """
        批量登记 chunk（向量入库成功后调用）。
        chunk_records: [(chunk_hash, file_id, chunk_index), ...]
        """
        _ensure_init()
        now = datetime.now(timezone.utc).isoformat()
        rows = [(h, fid, idx, now) for h, fid, idx in chunk_records]
        with _conn() as conn:
            conn.executemany(
                "INSERT OR IGNORE INTO chunks (id, file_id, chunk_index, created_at) VALUES (?, ?, ?, ?)",
                rows,
            )
        logger.debug("[Dedup] 登记 %d 个 chunk", len(rows))

    # ------------------------------------------------------------------
    #  查询 / 管理
    # ------------------------------------------------------------------

    @classmethod
    def get_file(cls, file_id: str) -> dict | None:
        _ensure_init()
        with _conn() as conn:
            row = conn.execute(
                "SELECT * FROM files WHERE id = ?", (file_id,)
            ).fetchone()
        return dict(row) if row else None

    @classmethod
    def list_files(cls, status: str | None = None) -> list[dict]:
        _ensure_init()
        with _conn() as conn:
            if status:
                rows = conn.execute(
                    "SELECT * FROM files WHERE status = ? ORDER BY created_at DESC", (status,)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM files ORDER BY created_at DESC"
                ).fetchall()
        return [dict(r) for r in rows]

    @classmethod
    def delete_file(cls, file_id: str) -> None:
        """删除文件及其 chunk 记录（用于重新处理）。"""
        _ensure_init()
        with _conn() as conn:
            conn.execute("DELETE FROM chunks WHERE file_id = ?", (file_id,))
            conn.execute("DELETE FROM files WHERE id = ?", (file_id,))
        logger.info("[Dedup] 删除记录: id=%s", file_id[:12])

    @classmethod
    def stats(cls) -> dict:
        _ensure_init()
        with _conn() as conn:
            total_files  = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
            done_files   = conn.execute("SELECT COUNT(*) FROM files WHERE status='done'").fetchone()[0]
            failed_files = conn.execute("SELECT COUNT(*) FROM files WHERE status='failed'").fetchone()[0]
            total_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        return {
            "total_files":  total_files,
            "done_files":   done_files,
            "failed_files": failed_files,
            "total_chunks": total_chunks,
        }
