"""
Ghi dữ liệu nạp vào vector DB.

Bảng liên quan (xem ``denso_backend/apps/rag/models.py``):

- ``rag_rawdata``   : tài liệu gốc. ``rag_worker`` CHỈ cập nhật (Django tạo row).
- ``rag_chunkdata`` : chunk + vector. ``rag_worker`` xoá và ghi lại (idempotent).

Lưu ý: ``created_at``/``updated_at`` là NOT NULL và do Django quản lý (không có
default ở DB), nên câu lệnh SQL ở đây phải tự set bằng ``now()``.
"""

from __future__ import annotations

import logging
import uuid
from typing import Dict, List, Sequence

from psycopg2.extras import Json, execute_values

from database.config import get_db

logger = logging.getLogger(__name__)

#: Thứ tự cột khi ghi chunk mới.
CHUNK_COLUMNS = (
    "id",
    "raw_data_id",
    "chunk_content",
    "chunk_index",
    "metadata",
    "vector",
    "fts",
    "created_at",
    "updated_at",
)


def to_vector_literal(vector: Sequence[float]) -> str:
    """Chuyển vector Python thành literal pgvector, ví dụ ``[0.1,-0.2]``."""
    return "[" + ",".join(repr(float(value)) for value in vector) + "]"


def set_raw_data_status(raw_data_id: str, status: str, error: str = "") -> None:
    """Cập nhật trạng thái nạp của tài liệu (``pending``/``processing``/...)."""
    query = (
        "UPDATE rag_rawdata "
        "SET status = %s, error = %s, updated_at = now() "
        "WHERE id = %s"
    )
    with get_db() as conn:
        with conn.cursor() as cursor:
            cursor.execute(query, (status, error[:4000], str(raw_data_id)))
            conn.commit()


def update_raw_data_content(raw_data_id: str, content: str) -> None:
    """Lưu nội dung đã bóc tách vào ``rag_rawdata.content``."""
    query = (
        "UPDATE rag_rawdata SET content = %s, updated_at = now() WHERE id = %s"
    )
    with get_db() as conn:
        with conn.cursor() as cursor:
            cursor.execute(query, (content, str(raw_data_id)))
            conn.commit()


def replace_chunks(
    raw_data_id: str,
    chunks: Sequence[Dict],
    vectors: Sequence[Sequence[float]],
) -> int:
    """Thay toàn bộ chunk của một tài liệu bằng danh sách mới.

    Xoá chunk cũ trước để nạp lại (re-ingest) không sinh dữ liệu trùng.

    Args:
        raw_data_id: UUID của ``RawData``.
        chunks: Danh sách chunk, mỗi phần tử có ``text`` và ``metadata``.
        vectors: Vector tương ứng, cùng thứ tự với ``chunks``.

    Returns:
        Số chunk đã ghi.

    Raises:
        ValueError: nếu độ dài ``chunks`` và ``vectors`` không khớp.
    """
    if len(chunks) != len(vectors):
        raise ValueError(
            f"Số chunk ({len(chunks)}) và số vector ({len(vectors)}) không khớp."
        )
    if not chunks:
        return 0

    template = "(%s, %s, %s, %s, %s, %s::vector, to_tsvector('simple', %s), now(), now())"
    rows = []
    for index, (chunk, vector) in enumerate(zip(chunks, vectors)):
        metadata = dict(chunk.get("metadata") or {})
        metadata.setdefault("chunk_index", index)
        text = chunk["text"]
        rows.append(
            (
                str(chunk.get("id") or uuid.uuid4()),
                str(raw_data_id),
                text,
                index,
                Json(metadata),
                to_vector_literal(vector),
                text,
            )
        )

    with get_db() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "DELETE FROM rag_chunkdata WHERE raw_data_id = %s",
                (str(raw_data_id),),
            )
            execute_values(
                cursor,
                f"INSERT INTO rag_chunkdata ({', '.join(CHUNK_COLUMNS)}) VALUES %s",
                rows,
                template=template,
                page_size=100,
            )
            conn.commit()
    logger.info("Đã ghi %d chunk cho RawData %s", len(rows), raw_data_id)
    return len(rows)


def count_chunks(raw_data_id: str) -> int:
    """Đếm số chunk hiện có của một tài liệu."""
    with get_db() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT count(*) FROM rag_chunkdata WHERE raw_data_id = %s",
                (str(raw_data_id),),
            )
            row = cursor.fetchone()
    return int(row[0]) if row else 0


def raw_data_exists(raw_data_id: str) -> bool:
    """Kiểm tra tài liệu có tồn tại trong ``rag_rawdata`` không."""
    with get_db() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT 1 FROM rag_rawdata WHERE id = %s", (str(raw_data_id),)
            )
            return cursor.fetchone() is not None


def get_vector_column_dimensions() -> int | None:
    """Số chiều của cột ``rag_chunkdata.vector`` theo khai báo trong DB.

    pgvector lưu số chiều trong ``pg_attribute.atttypmod``; trả về ``None``
    nếu bảng/cột chưa tồn tại.
    """
    query = (
        "SELECT atttypmod FROM pg_attribute "
        "WHERE attrelid = 'rag_chunkdata'::regclass AND attname = 'vector'"
    )
    try:
        with get_db() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query)
                row = cursor.fetchone()
    except Exception:  # noqa: BLE001 - bảng chưa tồn tại
        return None

    if not row or row[0] is None or row[0] <= 0:
        return None
    return int(row[0])


def ensure_schema_ready() -> List[str]:
    """Kiểm tra các cột/bảng cần thiết đã tồn tại.

    Dùng cho endpoint ``/ready`` để phát hiện sớm việc chưa chạy migration.
    """
    missing: List[str] = []
    with get_db() as conn:
        with conn.cursor() as cursor:
            for table, columns in (
                ("rag_rawdata", ("id", "content", "status", "error")),
                (
                    "rag_chunkdata",
                    ("id", "raw_data_id", "chunk_content", "vector", "metadata"),
                ),
            ):
                cursor.execute(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = %s",
                    (table,),
                )
                existing = {row[0] for row in cursor.fetchall()}
                if not existing:
                    missing.append(table)
                    continue
                for column in columns:
                    if column not in existing:
                        missing.append(f"{table}.{column}")
    return missing
