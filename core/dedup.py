"""
core/dedup.py — 去重工具类。

数据库表结构
-----------
companies 表：
    id          TEXT PRIMARY KEY
    name        TEXT
    created_at  TEXT

files 表：
    id                  TEXT PRIMARY KEY   -- 文件内容 SHA256
    filename            TEXT
    file_path           TEXT
    company_id          TEXT
    doc_type            TEXT               -- 文件类型（自动推断或人工确认）
    doc_type_confirmed  INTEGER DEFAULT 0  -- 0=自动推断 1=人工已确认
    status              TEXT               -- pending / done / failed
    created_at          TEXT

chunks 表：
    id          TEXT PRIMARY KEY
    file_id     TEXT
    chunk_index INTEGER
    created_at  TEXT
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


class DocType:
    LICENSE   = "license"
    INVOICE   = "invoice"
    TABLE     = "table"
    NAMEPLATE = "nameplate"
    DOCUMENT  = "document"
    UNKNOWN   = "unknown"

    ALL = {LICENSE, INVOICE, TABLE, NAMEPLATE, DOCUMENT, UNKNOWN}

    OCR_ENGINE: dict[str, str] = {
        LICENSE:   "ppocr",
        INVOICE:   "ppocr",
        TABLE:     "ppocr",
        NAMEPLATE: "got_ocr",
        DOCUMENT:  "ppocr",
        UNKNOWN:   "got_ocr",
    }


_initialized: bool = False


def _get_db_path() -> str:
    return settings.db_path


def _ensure_init() -> None:
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
                id                  TEXT PRIMARY KEY,
                filename            TEXT NOT NULL,
                file_path           TEXT NOT NULL,
                company_id          TEXT,
                doc_type            TEXT NOT NULL DEFAULT 'unknown',
                doc_type_confirmed  INTEGER NOT NULL DEFAULT 0,
                status              TEXT NOT NULL DEFAULT 'pending',
                created_at          TEXT NOT NULL,
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
        # 兼容旧库：如果 doc_type_confirmed 列不存在则补加
        try:
            conn.execute("ALTER TABLE files ADD COLUMN doc_type_confirmed INTEGER NOT NULL DEFAULT 0")
            logger.info("[Dedup] 已为旧库补加 doc_type_confirmed 列")
        except Exception:
            pass  # 列已存在，忽略
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


class Dedup:

    @staticmethod
    def _get_conn_ctx():
        """暴露 _conn 上下文管理器供需要直接操作数据库的场景使用。"""
        return _conn()

    # ------------------------------------------------------------------
    #  Hash
    # ------------------------------------------------------------------

    @staticmethod
    def hash_file(file_path: str | Path) -> str:
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    @staticmethod
    def hash_text(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    #  企业管理
    # ------------------------------------------------------------------

    @classmethod
    def register_company(cls, company_id: str, name: str) -> None:
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
        doc_type_confirmed: bool = False,
    ) -> str:
        _ensure_init()
        path = Path(file_path)
        file_id = cls.hash_file(path)
        fname = filename or path.name
        dt = doc_type if doc_type in DocType.ALL else DocType.UNKNOWN
        confirmed = 1 if doc_type_confirmed else 0
        now = datetime.now(timezone.utc).isoformat()

        with _conn() as conn:
            conn.execute(
                """
                INSERT INTO files (id, filename, file_path, company_id, doc_type,
                                   doc_type_confirmed, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, 'pending', ?)
                ON CONFLICT(id) DO UPDATE SET
                    status              = 'pending',
                    file_path           = excluded.file_path,
                    filename            = excluded.filename,
                    company_id          = excluded.company_id,
                    doc_type            = excluded.doc_type,
                    doc_type_confirmed  = excluded.doc_type_confirmed
                """,
                (file_id, fname, str(path), company_id, dt, confirmed, now),
            )
        logger.info(
            "[Dedup] 登记文件: %s  company=%s  type=%s  confirmed=%s  id=%s",
            fname, company_id, dt, confirmed, file_id[:12],
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
    #  doc_type 人工确认
    # ------------------------------------------------------------------

    @classmethod
    def update_doc_type(
        cls,
        file_id: str,
        doc_type: str,
        confirmed: bool = True,
    ) -> bool:
        """
        人工修改文件的 doc_type。

        Parameters
        ----------
        file_id   : 文件 SHA256
        doc_type  : 新的文件类型
        confirmed : True=人工确认，False=重置为自动推断状态

        Returns
        -------
        True=更新成功，False=文件不存在
        """
        _ensure_init()
        if doc_type not in DocType.ALL:
            raise ValueError(f"无效的 doc_type: {doc_type}")

        with _conn() as conn:
            row = conn.execute(
                "SELECT id FROM files WHERE id = ?", (file_id,)
            ).fetchone()
            if not row:
                return False
            conn.execute(
                """
                UPDATE files
                SET doc_type = ?, doc_type_confirmed = ?
                WHERE id = ?
                """,
                (doc_type, 1 if confirmed else 0, file_id),
            )
        logger.info(
            "[Dedup] 更新 doc_type: id=%s  type=%s  confirmed=%s",
            file_id[:12], doc_type, confirmed,
        )
        return True

    # ------------------------------------------------------------------
    #  Chunk 级去重
    # ------------------------------------------------------------------

    @classmethod
    def filter_new_chunks(
        cls,
        chunks: list[Document],
        file_id: str,
    ) -> tuple[list[Document], list[str]]:
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
        _ensure_init()
        now = datetime.now(timezone.utc).isoformat()
        rows = [(h, fid, idx, now) for h, fid, idx in chunk_records]
        with _conn() as conn:
            conn.executemany(
                "INSERT OR IGNORE INTO chunks (id, file_id, chunk_index, created_at) VALUES (?, ?, ?, ?)",
                rows,
            )
        logger.debug("[Dedup] 登记 %d 个 chunk", len(rows))

    @classmethod
    def delete_chunks_by_file(cls, file_id: str) -> int:
        """删除某文件的全部 chunk 记录，返回删除数量。"""
        _ensure_init()
        with _conn() as conn:
            n = conn.execute(
                "DELETE FROM chunks WHERE file_id = ?", (file_id,)
            ).rowcount
        logger.info("[Dedup] 删除 chunk 记录: file_id=%s  共 %d 条", file_id[:12], n)
        return n

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
        _ensure_init()
        with _conn() as conn:
            conn.execute("DELETE FROM chunks WHERE file_id = ?", (file_id,))
            conn.execute("DELETE FROM files WHERE id = ?", (file_id,))
        logger.info("[Dedup] 删除记录: id=%s", file_id[:12])

    @classmethod
    def delete_company(cls, company_id: str) -> int:
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
            confirmed_files = conn.execute("SELECT COUNT(*) FROM files WHERE doc_type_confirmed=1").fetchone()[0]
            total_chunks    = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            type_rows       = conn.execute(
                "SELECT doc_type, COUNT(*) as cnt FROM files GROUP BY doc_type"
            ).fetchall()
        return {
            "total_companies":  total_companies,
            "total_files":      total_files,
            "done_files":       done_files,
            "failed_files":     failed_files,
            "confirmed_files":  confirmed_files,
            "total_chunks":     total_chunks,
            "by_doc_type":      {r["doc_type"]: r["cnt"] for r in type_rows},
        }