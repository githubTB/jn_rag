"""
core/dedup.py — 去重工具类。

数据库表结构
-----------
companies 表：
    id          TEXT PRIMARY KEY   -- 企业唯一 ID（业务方生成或 UUID）
    name        TEXT               -- 企业名称
    created_at  TEXT

files 表：
    id          TEXT PRIMARY KEY   -- 文件内容 SHA256
    filename    TEXT
    file_path   TEXT
    company_id  TEXT               -- 所属企业 ID（可为空，单文件场景）
    doc_type    TEXT               -- 文件类型：license/invoice/table/nameplate/document/unknown
    status      TEXT               -- pending / done / failed
    created_at  TEXT

chunks 表：
    id          TEXT PRIMARY KEY   -- chunk 文本 SHA256
    file_id     TEXT
    chunk_index INTEGER
    created_at  TEXT

使用示例
--------
    from core.dedup import Dedup

    # 登记企业
    Dedup.register_company("company_001", "澳龙生物科技有限公司")

    # 登记文件（带企业ID和文件类型）
    file_id = Dedup.register_file(
        "uploads/营业执照.jpg",
        company_id="company_001",
        doc_type="license",
    )

    # chunk 去重
    new_chunks, hashes = Dedup.filter_new_chunks(chunks, file_id)

    # 完成
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
# 文件类型常量
# ---------------------------------------------------------------------------

class DocType:
    LICENSE   = "license"    # 营业执照、证件
    INVOICE   = "invoice"    # 发票、单据
    TABLE     = "table"      # 表格、报表
    NAMEPLATE = "nameplate"  # 设备铭牌、现场照片
    DOCUMENT  = "document"   # Word/PDF 文档
    UNKNOWN   = "unknown"    # 未知，兜底

    ALL = {LICENSE, INVOICE, TABLE, NAMEPLATE, DOCUMENT, UNKNOWN}

    # 各类型对应 OCR 引擎（给 ImageExtractor 路由用）
    OCR_ENGINE: dict[str, str] = {
        LICENSE:   "ppocr",      # PP-OCRv4，CPU 轻量
        INVOICE:   "ppocr",
        TABLE:     "ppocr",      # PP-OCRv4 + TableRec
        NAMEPLATE: "got_ocr",    # GOT-OCR2，复杂场景
        DOCUMENT:  "ppocr",
        UNKNOWN:   "got_ocr",    # 不确定走 GOT-OCR2 兜底
    }


# ---------------------------------------------------------------------------
# 模块级数据库单例
# ---------------------------------------------------------------------------

_initialized: bool = False


def _get_db_path() -> str:
    return settings.db_path


def _ensure_init() -> None:
    """建表（幂等，重复调用安全）。"""
    global _initialized
    if _initialized:
        return
    with _conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS companies (
                id          TEXT PRIMARY KEY,
                name        TEXT NOT NULL,
                created_at  TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS files (
                id          TEXT PRIMARY KEY,
                filename    TEXT NOT NULL,
                file_path   TEXT NOT NULL,
                company_id  TEXT,
                doc_type    TEXT NOT NULL DEFAULT 'unknown',
                status      TEXT NOT NULL DEFAULT 'pending',
                created_at  TEXT NOT NULL,
                FOREIGN KEY (company_id) REFERENCES companies(id)
            );

            CREATE TABLE IF NOT EXISTS chunks (
                id          TEXT PRIMARY KEY,
                file_id     TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                created_at  TEXT NOT NULL,
                FOREIGN KEY (file_id) REFERENCES files(id)
            );

            CREATE INDEX IF NOT EXISTS idx_files_company_id ON files(company_id);
            CREATE INDEX IF NOT EXISTS idx_files_doc_type   ON files(doc_type);
            CREATE INDEX IF NOT EXISTS idx_chunks_file_id   ON chunks(file_id);
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
        Dedup.register_company(company_id, name)
        file_id = Dedup.register_file(path, company_id=..., doc_type=...)
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
    #  企业管理
    # ------------------------------------------------------------------

    @classmethod
    def register_company(cls, company_id: str, name: str) -> None:
        """
        登记企业（幂等，已存在则更新名称）。

        Parameters
        ----------
        company_id : 业务方定义的企业唯一 ID（如统一社会信用代码或自定义 ID）
        name       : 企业名称
        """
        _ensure_init()
        now = datetime.now(timezone.utc).isoformat()
        with _conn() as conn:
            conn.execute(
                """
                INSERT INTO companies (id, name, created_at) VALUES (?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET name = excluded.name
                """,
                (company_id, name, now),
            )
        logger.info("[Dedup] 登记企业: %s (%s)", name, company_id)

    @classmethod
    def get_company(cls, company_id: str) -> dict | None:
        _ensure_init()
        with _conn() as conn:
            row = conn.execute(
                "SELECT * FROM companies WHERE id = ?", (company_id,)
            ).fetchone()
        return dict(row) if row else None

    @classmethod
    def list_companies(cls) -> list[dict]:
        _ensure_init()
        with _conn() as conn:
            rows = conn.execute(
                "SELECT * FROM companies ORDER BY created_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    #  文件级去重
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
        company_id: str | None = None,
        doc_type: str = DocType.UNKNOWN,
    ) -> str:
        """
        登记文件，返回 file_id（SHA256）。
        已存在时重置为 pending（支持重试）。

        Parameters
        ----------
        file_path  : 磁盘路径
        filename   : 显示文件名（None 时取 path.name）
        company_id : 所属企业 ID（可为 None）
        doc_type   : 文件类型，见 DocType 常量
        """
        _ensure_init()
        path = Path(file_path)
        file_id = cls.hash_file(path)
        fname = filename or path.name
        dt = doc_type if doc_type in DocType.ALL else DocType.UNKNOWN
        now = datetime.now(timezone.utc).isoformat()

        with _conn() as conn:
            conn.execute(
                """
                INSERT INTO files (id, filename, file_path, company_id, doc_type, status, created_at)
                VALUES (?, ?, ?, ?, ?, 'pending', ?)
                ON CONFLICT(id) DO UPDATE SET
                    status     = 'pending',
                    file_path  = excluded.file_path,
                    filename   = excluded.filename,
                    company_id = excluded.company_id,
                    doc_type   = excluded.doc_type
                """,
                (file_id, fname, str(path), company_id, dt, now),
            )
        logger.info(
            "[Dedup] 登记文件: %s  company=%s  type=%s  id=%s",
            fname, company_id, dt, file_id[:12],
        )
        return file_id

    @classmethod
    def mark_done(cls, file_id: str) -> None:
        _ensure_init()
        with _conn() as conn:
            conn.execute(
                "UPDATE files SET status = 'done' WHERE id = ?", (file_id,)
            )
        logger.info("[Dedup] 文件完成: id=%s", file_id[:12])

    @classmethod
    def mark_failed(cls, file_id: str) -> None:
        _ensure_init()
        with _conn() as conn:
            conn.execute(
                "UPDATE files SET status = 'failed' WHERE id = ?", (file_id,)
            )
        logger.warning("[Dedup] 文件失败: id=%s", file_id[:12])

    # ------------------------------------------------------------------
    #  Chunk 级去重
    # ------------------------------------------------------------------

    @classmethod
    def filter_new_chunks(
        cls,
        chunks: list[Document],
        file_id: str,
    ) -> tuple[list[Document], list[str]]:
        """过滤已入库 chunk，返回 (新chunk列表, 对应hash列表)。"""
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
        """批量登记 chunk。chunk_records: [(chunk_hash, file_id, chunk_index), ...]"""
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
    #  查询
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
    def list_files(
        cls,
        status: str | None = None,
        company_id: str | None = None,
        doc_type: str | None = None,
    ) -> list[dict]:
        """列出文件，支持按状态、企业、类型过滤。"""
        _ensure_init()
        conditions = []
        params = []
        if status:
            conditions.append("status = ?")
            params.append(status)
        if company_id:
            conditions.append("company_id = ?")
            params.append(company_id)
        if doc_type:
            conditions.append("doc_type = ?")
            params.append(doc_type)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        with _conn() as conn:
            rows = conn.execute(
                f"SELECT * FROM files {where} ORDER BY created_at DESC", params
            ).fetchall()
        return [dict(r) for r in rows]

    @classmethod
    def delete_file(cls, file_id: str) -> None:
        """删除文件及其 chunk 记录。"""
        _ensure_init()
        with _conn() as conn:
            conn.execute("DELETE FROM chunks WHERE file_id = ?", (file_id,))
            conn.execute("DELETE FROM files WHERE id = ?", (file_id,))
        logger.info("[Dedup] 删除记录: id=%s", file_id[:12])

    @classmethod
    def delete_company(cls, company_id: str) -> int:
        """
        删除企业及其所有文件记录（Milvus 向量需另外清理）。
        返回删除的文件数。
        """
        _ensure_init()
        files = cls.list_files(company_id=company_id)
        for f in files:
            cls.delete_file(f["id"])
        with _conn() as conn:
            conn.execute("DELETE FROM companies WHERE id = ?", (company_id,))
        logger.info("[Dedup] 删除企业: %s，共 %d 个文件", company_id, len(files))
        return len(files)

    @classmethod
    def stats(cls) -> dict:
        _ensure_init()
        with _conn() as conn:
            total_companies = conn.execute("SELECT COUNT(*) FROM companies").fetchone()[0]
            total_files     = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
            done_files      = conn.execute("SELECT COUNT(*) FROM files WHERE status='done'").fetchone()[0]
            failed_files    = conn.execute("SELECT COUNT(*) FROM files WHERE status='failed'").fetchone()[0]
            total_chunks    = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            # 按类型统计
            type_rows = conn.execute(
                "SELECT doc_type, COUNT(*) as cnt FROM files GROUP BY doc_type"
            ).fetchall()
        return {
            "total_companies": total_companies,
            "total_files":     total_files,
            "done_files":      done_files,
            "failed_files":    failed_files,
            "total_chunks":    total_chunks,
            "by_doc_type":     {r["doc_type"]: r["cnt"] for r in type_rows},
        }
        