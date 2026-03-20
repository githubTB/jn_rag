"""
scripts/cleanup.py — RAG 知识库人工清理工具。

用法
----
    python scripts/cleanup.py list                    # 列出所有文件
    python scripts/cleanup.py list --status failed    # 只看失败的
    python scripts/cleanup.py stats                   # 统计数据
    python scripts/cleanup.py delete <file_id>        # 删除指定文件
    python scripts/cleanup.py delete --name 营业执照  # 按文件名模糊删除
    python scripts/cleanup.py retry                   # 重置所有 failed 为 pending
    python scripts/cleanup.py retry <file_id>         # 重置指定文件
    python scripts/cleanup.py drop-collection         # 删除 Milvus 集合（重建用）
    python scripts/cleanup.py purge                   # 清空全部数据（危险）
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from core.dedup import Dedup, _conn, _ensure_init


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _milvus_connect():
    from pymilvus import connections
    connections.connect(host=settings.milvus_host, port=settings.milvus_port)


def _drop_milvus_collection(col_name: str) -> bool:
    try:
        from pymilvus import utility
        _milvus_connect()
        if utility.has_collection(col_name):
            utility.drop_collection(col_name)
            import core.embedder as emb
            emb._collection = None          # 重置单例，下次重新建
            print(f"  ✓ Milvus 集合 '{col_name}' 已删除")
            return True
        else:
            print(f"  ! Milvus 集合 '{col_name}' 不存在，跳过")
            return False
    except Exception as e:
        print(f"  ! Milvus 操作失败: {e}")
        return False


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------

def cmd_list(args):
    files = Dedup.list_files(status=args.status)
    if not files:
        print("（无记录）")
        return

    icon = {"done": "✓", "failed": "✗", "pending": "…"}
    print(f"\n{'':2}{'状态':<8}{'文件名':<42}{'file_id':<18}{'时间'}")
    print("─" * 84)
    for f in files:
        ic  = icon.get(f["status"], "?")
        fid = f["id"][:16] + ".."
        ts  = f["created_at"][:19].replace("T", " ")
        print(f" {ic} {f['status']:<7}{f['filename'][:40]:<42}{fid}  {ts}")
    print(f"\n共 {len(files)} 条")


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------

def cmd_stats(args):
    s = Dedup.stats()
    milvus_count = "（未连接）"
    try:
        from core.embedder import Embedder
        milvus_count = str(Embedder.count())
    except Exception as e:
        milvus_count = f"错误: {e}"

    print("\n─── 库内统计 ──────────────────────────────")
    print(f"  文件总数       : {s['total_files']}")
    print(f"    已完成       : {s['done_files']}")
    print(f"    失败         : {s['failed_files']}")
    print(f"    处理中       : {s['total_files'] - s['done_files'] - s['failed_files']}")
    print(f"  chunk 记录数   : {s['total_chunks']}  (SQLite)")
    print(f"  向量数         : {milvus_count}  (Milvus)")
    print(f"  SQLite 路径    : {settings.db_path}")
    print(f"  Milvus 集合    : {settings.milvus_collection}")
    print("────────────────────────────────────────────\n")


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------

def cmd_delete(args):
    if args.file_id:
        _delete_one(args.file_id)
    elif args.name:
        files = Dedup.list_files()
        matched = [f for f in files if args.name in f["filename"]]
        if not matched:
            print(f"未找到匹配文件名: {args.name}")
            return
        print(f"找到 {len(matched)} 个匹配文件：")
        for f in matched:
            print(f"  {f['id'][:16]}..  {f['filename']}  [{f['status']}]")
        confirm = input(f"\n确认删除以上 {len(matched)} 个？[y/N] ").strip().lower()
        if confirm != "y":
            print("已取消")
            return
        for f in matched:
            _delete_one(f["id"])
    else:
        print("请指定 file_id 或 --name 文件名关键字")


def _delete_one(file_id: str):
    """支持传完整 id 或前缀。"""
    record = Dedup.get_file(file_id)
    if not record:
        # 前缀匹配
        all_files = Dedup.list_files()
        matched = [f for f in all_files if f["id"].startswith(file_id)]
        if len(matched) == 1:
            record = matched[0]
            file_id = record["id"]
        elif len(matched) > 1:
            print(f"前缀匹配到多个文件，请提供更长的 file_id：")
            for f in matched:
                print(f"  {f['id'][:20]}  {f['filename']}")
            return
        else:
            print(f"未找到: {file_id}")
            return

    print(f"\n删除: {record['filename']}  ({file_id[:16]}..) [{record['status']}]")

    # 1. 删 Milvus 向量
    try:
        from core.embedder import Embedder
        Embedder.delete_by_file(file_id)
        print("  ✓ Milvus 向量已删除")
    except Exception as e:
        print(f"  ! Milvus 删除失败: {e}")

    # 2. 删 SQLite 记录（文件 + chunk）
    Dedup.delete_file(file_id)
    print("  ✓ SQLite 记录已删除")


# ---------------------------------------------------------------------------
# retry
# ---------------------------------------------------------------------------

def cmd_retry(args):
    _ensure_init()
    if args.file_id:
        record = Dedup.get_file(args.file_id)
        if not record:
            print(f"未找到: {args.file_id}")
            return
        with _conn() as conn:
            conn.execute(
                "UPDATE files SET status='pending' WHERE id=?", (args.file_id,)
            )
        print(f"✓ 已重置为 pending: {record['filename']}")
    else:
        with _conn() as conn:
            n = conn.execute(
                "UPDATE files SET status='pending' WHERE status='failed'"
            ).rowcount
        print(f"✓ 已重置 {n} 个 failed 文件为 pending，下次调用入库接口会重新处理")


# ---------------------------------------------------------------------------
# drop-collection（只删 Milvus，SQLite 保留）
# ---------------------------------------------------------------------------

def cmd_drop_collection(args):
    col_name = settings.milvus_collection
    print(f"\n将删除 Milvus 集合: {col_name}")
    print("注意：SQLite 的文件/chunk 记录保留，只清空向量数据。")
    confirm = input("确认？[y/N] ").strip().lower()
    if confirm != "y":
        print("已取消")
        return
    _drop_milvus_collection(col_name)
    print(f"\n完成。下次调用入库接口时集合会自动重建。")
    print(f"如需重新入库已有文件，执行：python scripts/cleanup.py retry")


# ---------------------------------------------------------------------------
# purge（清空全部）
# ---------------------------------------------------------------------------

def cmd_purge(args):
    s = Dedup.stats()
    print(f"\n⚠  即将清空全部数据（不可恢复）：")
    print(f"  SQLite 文件记录  : {s['total_files']} 条")
    print(f"  SQLite chunk记录 : {s['total_chunks']} 条")
    try:
        from core.embedder import Embedder
        print(f"  Milvus 向量      : {Embedder.count()} 条")
    except Exception:
        print(f"  Milvus           : 无法连接")

    confirm = input("\n输入 'yes' 确认清空: ").strip()
    if confirm != "yes":
        print("已取消")
        return

    # 1. 删 Milvus 集合
    _drop_milvus_collection(settings.milvus_collection)

    # 2. 清 SQLite
    _ensure_init()
    with _conn() as conn:
        conn.execute("DELETE FROM chunks")
        conn.execute("DELETE FROM files")
    print("  ✓ SQLite 记录已清空")
    print("\n清理完成。")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="RAG 知识库人工清理工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # list
    p_list = sub.add_parser("list", help="列出文件记录")
    p_list.add_argument("--status", choices=["done", "failed", "pending"], help="按状态过滤")

    # stats
    sub.add_parser("stats", help="统计数据")

    # delete
    p_del = sub.add_parser("delete", help="删除文件")
    p_del.add_argument("file_id", nargs="?", help="文件 SHA256（支持前缀）")
    p_del.add_argument("--name", help="文件名关键字（模糊匹配）")

    # retry
    p_retry = sub.add_parser("retry", help="重置 failed 文件为 pending")
    p_retry.add_argument("file_id", nargs="?", help="指定 file_id，不填则重置全部 failed")

    # drop-collection
    sub.add_parser("drop-collection", help="删除 Milvus 集合（保留 SQLite，重建用）")

    # purge
    sub.add_parser("purge", help="清空全部数据（危险）")

    args = parser.parse_args()
    {
        "list":            cmd_list,
        "stats":           cmd_stats,
        "delete":          cmd_delete,
        "retry":           cmd_retry,
        "drop-collection": cmd_drop_collection,
        "purge":           cmd_purge,
    }[args.cmd](args)


if __name__ == "__main__":
    main()