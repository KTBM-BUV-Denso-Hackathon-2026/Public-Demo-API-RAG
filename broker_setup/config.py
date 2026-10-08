"""
Khởi tạo Celery app cho ``rag_worker``.

Worker chạy bằng:

    celery -A main:celery_app worker -Q rag -l info --pool=solo

(``--pool=solo`` bắt buộc trên Windows vì prefork không hoạt động ổn định.)
"""

import logging

from celery import Celery

import settings

logger = logging.getLogger(__name__)

celery_app = Celery(
    "rag_worker",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND or None,
)

celery_app.conf.update(
    task_default_queue=settings.CELERY_TASK_QUEUE,
    task_ignore_result=True,
    task_track_started=True,
    # Không nhận task quá khứ khi worker restart.
    worker_prefetch_multiplier=1,
    broker_connection_retry_on_startup=True,
)

#: Tên cũ, giữ lại để không phá vỡ import cũ.
message_broker = celery_app
