"""
Đường ống nạp dữ liệu: file → text → chunk → vector → ``rag_chunkdata``.

Đây là **nguồn sự thật duy nhất** cho việc nạp dữ liệu; cả API ``POST /ingest``
lẫn task Celery ``process.rag.task`` đều gọi ``ingest_document``.
"""

from __future__ import annotations

import logging
import time
from typing import Dict

import settings
from database import controllers as db
from embedding_engine.controllers import chunks_process
from utilities.data_preprocessor import content_extractor, split_text
from utilities.files_handler import delete_file, fetch_document

logger = logging.getLogger(__name__)

STATUS_PROCESSING = "processing"
STATUS_DONE = "done"
STATUS_FAILED = "failed"


def _assert_dimension_matches_db() -> None:
    """Chặn sớm khi ``EMBEDDING_DIM`` lệch với cột ``vector`` trong DB.

    Không có bước này, một tài liệu lệch cấu hình sẽ đi trọn đường (tải file →
    đọc nội dung → nạp model embedding, mất hàng chục giây) rồi mới nổ lỗi SQL
    ``expected 768 dimensions, not 384`` ngay lúc INSERT. Ở đây ta fail trước
    khi tốn công.

    Raises:
        RuntimeError: khi đọc được cột ``vector`` nhưng số chiều không khớp.
    """
    column_dim = db.get_vector_column_dimensions()
    if column_dim is None:
        # Bảng/cột chưa tồn tại -> để ``/ready`` và migration báo, không chặn ở đây.
        logger.warning(
            "Không đọc được số chiều cột rag_chunkdata.vector (chưa migrate?); "
            "bỏ qua kiểm tra EMBEDDING_DIM."
        )
        return
    if column_dim != settings.EMBEDDING_DIM:
        raise RuntimeError(
            f"Sai số chiều embedding: cột rag_chunkdata.vector là vector({column_dim}) "
            f"nhưng EMBEDDING_DIM={settings.EMBEDDING_DIM}. Sửa EMBEDDING_DIM trong "
            f"rag_worker/.env (và RAG_EMBEDDING_DIM + migration của denso_backend) cho khớp."
        )


def ingest_document(raw_data_id: str, file_url: str) -> Dict:
    """Nạp một tài liệu vào vector DB.

    Các bước:
    1. Kiểm tra ``EMBEDDING_DIM`` có khớp cột ``vector`` trong DB không (fail sớm).
    2. Đánh dấu ``processing``.
    3. Lấy file (tải qua HTTP hoặc đọc file local).
    4. Bóc tách nội dung → lưu vào ``rag_rawdata.content``.
    5. Chia chunk → vector hoá theo batch.
    6. Xoá chunk cũ và ghi chunk mới vào ``rag_chunkdata``.
    7. Đánh dấu ``done`` (hoặc ``failed`` kèm thông báo lỗi).

    Args:
        raw_data_id: UUID của ``RawData`` (do denso_backend tạo).
        file_url: URL hoặc đường dẫn tới file cần nạp.

    Returns:
        dict kết quả: ``raw_data_id``, ``status``, ``chunks``, ``characters``,
        ``elapsed_ms``.

    Raises:
        Exception: lỗi gốc được ném lại sau khi đã ghi ``status=failed`` vào DB.
    """
    started = time.time()
    logger.info("Bắt đầu nạp RawData %s từ %s", raw_data_id, file_url)

    try:
        if not db.raw_data_exists(raw_data_id):
            raise ValueError(f"Không tìm thấy RawData {raw_data_id} trong DB.")

        # Fail sớm trước khi tốn thời gian tải file + nạp model embedding.
        _assert_dimension_matches_db()

        db.set_raw_data_status(raw_data_id, STATUS_PROCESSING)

        file_path, should_delete = fetch_document(file_url)
        try:
            content = content_extractor(file_path)
        finally:
            if should_delete:
                delete_file(file_path)

        if not content.strip():
            raise ValueError("File rỗng hoặc không bóc tách được nội dung text.")

        db.update_raw_data_content(raw_data_id, content)

        chunks = split_text(content, source=file_url)
        if not chunks:
            raise ValueError("Không tạo được chunk nào từ tài liệu.")

        vectors = chunks_process(chunks)
        written = db.replace_chunks(raw_data_id, chunks, vectors)
        db.set_raw_data_status(raw_data_id, STATUS_DONE)

        elapsed_ms = int((time.time() - started) * 1000)
        logger.info(
            "Nạp xong RawData %s: %d chunk trong %d ms",
            raw_data_id,
            written,
            elapsed_ms,
        )
        return {
            "raw_data_id": str(raw_data_id),
            "status": STATUS_DONE,
            "chunks": written,
            "characters": len(content),
            "elapsed_ms": elapsed_ms,
        }
    except Exception as exc:  # noqa: BLE001 - ghi lại nguyên nhân rồi ném tiếp
        logger.exception("Nạp dữ liệu thất bại cho RawData %s", raw_data_id)
        message = f"{type(exc).__name__}: {exc}"
        try:
            db.set_raw_data_status(raw_data_id, STATUS_FAILED, message)
        except Exception:  # pragma: no cover - DB có thể đang hỏng
            logger.exception("Không ghi được trạng thái failed cho %s", raw_data_id)
        raise
