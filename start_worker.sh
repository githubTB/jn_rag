#!/bin/bash
# start_worker.sh — 启动 Celery Worker

cd /Users/david/data/www/python/jn_rag

echo "启动 Celery Worker..."
echo "Broker: redis://localhost:6379/0"
echo "队列: ingest"
echo ""

export PYTHONPATH=$(pwd)

# 激活虚拟环境
source venv/bin/activate

# 启动 Worker
# --concurrency=2  同时处理2个任务（OCR是IO密集型，2个并发合适）
# --loglevel=info  日志级别
# -Q ingest        只消费 ingest 队列
celery -A celery_app worker \
    --loglevel=info \
    --concurrency=2 \
    --pool=solo \
    -Q ingest \
    --logfile=logs/celery_worker.log \
    --pidfile=logs/celery_worker.pid \
    --detach


echo "Worker 已在后台启动"
echo "查看日志: tail -f logs/celery_worker.log"
echo "停止 Worker: celery -A celery_app control shutdown"