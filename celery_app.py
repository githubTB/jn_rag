"""
celery_app.py — Celery 实例配置。

启动 Worker：
    celery -A celery_app worker --loglevel=info --concurrency=2 -Q ingest

查看任务状态：
    celery -A celery_app inspect active
    celery -A celery_app inspect reserved
"""

from celery import Celery
from config.settings import settings

app = Celery(
    "jn_rag",
    broker=settings.redis_broker_url,
    backend=settings.redis_backend_url,
)

app.conf.update(
    # 序列化
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],

    # 时区
    timezone="Asia/Shanghai",
    enable_utc=True,

    # 任务超时：单个文件最多 10 分钟
    task_soft_time_limit=3600,
    task_time_limit=3900,

    # 结果保留 24 小时
    result_expires=86400,

    # Worker 每次预取 1 个任务，避免大任务堆积
    worker_prefetch_multiplier=1,
    task_acks_late=True,

    # 队列定义
    task_routes={
        "core.tasks_celery.ingest_file": {"queue": "ingest"},
        "core.tasks_celery.ingest_batch": {"queue": "ingest"},
    },

    # 失败重试
    task_max_retries=3,
    task_default_retry_delay=10,
)

# 自动发现任务
app.autodiscover_tasks(["core.tasks_celery"])