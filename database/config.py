"""
Kết nối tới PostgreSQL + pgvector (CHUNG với ``denso_backend``).

Hệ thống dùng **một** database duy nhất cho cả bảng nghiệp vụ lẫn bảng vector
(lý do: ``rag_rawdata.user_id`` là FK tới ``core_users``, PostgreSQL không hỗ trợ
FK xuyên database). Vì vậy chỉ cần đúng một biến ``DATABASE_URL`` — phải trỏ
vào **cùng** DB mà ``denso_backend`` dùng, nếu không worker sẽ ghi chunk vào
một DB mà không ai đọc.

Dùng connection pool để tái sử dụng kết nối giữa các task.
"""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from typing import Iterator, Optional

import psycopg2
from psycopg2 import pool as psycopg2_pool

import settings

logger = logging.getLogger(__name__)

_pool: Optional[psycopg2_pool.ThreadedConnectionPool] = None
_pool_lock = threading.Lock()


def _database_url() -> str:
    """URL kết nối DB; báo lỗi rõ ràng nếu chưa cấu hình."""
    url = settings.DATABASE_URL
    if not url:
        raise RuntimeError(
            "Chưa cấu hình DATABASE_URL cho rag_worker. "
            "Phải trỏ TỚI CÙNG database với denso_backend. "
            "Xem rag_worker/.env.example."
        )
    return url


def get_pool() -> psycopg2_pool.ThreadedConnectionPool:
    """Trả về connection pool dùng chung (khởi tạo lần đầu khi cần)."""
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                _pool = psycopg2_pool.ThreadedConnectionPool(
                    minconn=settings.DB_POOL_MIN,
                    maxconn=settings.DB_POOL_MAX,
                    dsn=_database_url(),
                )
                logger.info(
                    "Đã tạo connection pool tới DB (min=%d, max=%d)",
                    settings.DB_POOL_MIN,
                    settings.DB_POOL_MAX,
                )
    return _pool


def close_pool() -> None:
    """Đóng toàn bộ kết nối trong pool (dùng khi shutdown/test)."""
    global _pool
    with _pool_lock:
        if _pool is not None:
            _pool.closeall()
            _pool = None


@contextmanager
def get_db() -> Iterator["psycopg2.extensions.connection"]:
    """Mượn một kết nối từ pool, tự trả lại sau khi dùng xong."""
    pool = get_pool()
    conn = pool.getconn()
    broken = False
    try:
        yield conn
    except Exception:
        broken = True
        try:
            conn.rollback()
        except Exception:  # pragma: no cover - kết nối đã chết
            pass
        raise
    finally:
        try:
            if broken or conn.closed:
                pool.putconn(conn, close=True)
            else:
                if conn.get_transaction_status() != psycopg2.extensions.TRANSACTION_STATUS_IDLE:
                    conn.rollback()
                pool.putconn(conn)
        except Exception:  # pragma: no cover - không được che lỗi gốc
            logger.exception("Lỗi khi trả kết nối về pool")
