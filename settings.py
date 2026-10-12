"""
Cấu hình tập trung cho microservice ``rag_worker``.

Đọc từ biến môi trường / file ``.env`` đặt cạnh ``main.py``.
Xem ``.env.example`` để biết danh sách đầy đủ.
"""

import os
from pathlib import Path

import dotenv

dotenv.load_dotenv()

BASE_DIR = Path(__file__).resolve().parent


def _get_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return default if value is None or value == "" else value


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None or value == "" else int(value)


def _get_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value is None or value == "" else float(value)


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


# ---------------------------------------------------------------------------
# PostgreSQL + pgvector — CHUNG với denso_backend và llm_api
# ---------------------------------------------------------------------------
DATABASE_URL: str = _get_str("DATABASE_URL", "")
DB_POOL_MIN: int = _get_int("DB_POOL_MIN", 1)
DB_POOL_MAX: int = _get_int("DB_POOL_MAX", 5)

# Số lần thử lại khi kết nối lấy từ pool đã bị pooler đóng bởi.
# Xem database/config.py::_borrow_live_connection để hiểu lý do.
DB_PREPING_ATTEMPTS: int = _get_int("DB_PREPING_ATTEMPTS", 2)

# ---------------------------------------------------------------------------
# Embedding — PHẢI khớp EMBEDDING_MODEL/EMBEDDING_DIM của llm_api
# và cột ``vector`` của bảng rag_chunkdata.
# ---------------------------------------------------------------------------
EMBEDDING_MODEL: str = _get_str(
    "EMBEDDING_MODEL", "bkai-foundation-models/vietnamese-bi-encoder"
)
EMBEDDING_DIM: int = _get_int("EMBEDDING_DIM", 768)
STRICT_EMBEDDING_DIM: bool = _get_bool("STRICT_EMBEDDING_DIM", True)
EMBEDDING_BATCH_SIZE: int = _get_int("EMBEDDING_BATCH_SIZE", 32)

# ---------------------------------------------------------------------------
# Chia nhỏ văn bản
# ---------------------------------------------------------------------------
#: Kích thước tối đa (ký tự) của một chunk.
CHUNK_SIZE: int = _get_int("CHUNK_SIZE", 1000)
#: Số ký tự tối thiểu để một mảnh văn bản được coi là chunk hợp lệ.
MIN_CHUNK_SIZE: int = _get_int("MIN_CHUNK_SIZE", 80)

# ---------------------------------------------------------------------------
# Tải file
# ---------------------------------------------------------------------------
DOWNLOAD_TIMEOUT: float = _get_float("DOWNLOAD_TIMEOUT", 60.0)
#: Thư mục chứa media của denso_backend, dùng khi ``file_url`` là đường dẫn nội bộ.
#: PHẢI trỏ đúng ``MEDIA_ROOT`` của denso_backend (mặc định
#: ``<workspace>/denso_backend/media``) — xem ``config/settings/base.py``:
#: ``MEDIA_ROOT = BASE_DIR / "media"`` và ``BASE_DIR`` = thư mục ``denso_backend/``.
MEDIA_ROOT: str = _get_str("MEDIA_ROOT", "")
#: Tiền tố URL media của Django (mặc định ``/media/``).
MEDIA_URL: str = _get_str("MEDIA_URL", "/media/")

# ---------------------------------------------------------------------------
# Xác thực nội bộ (xem security.py)
# ---------------------------------------------------------------------------
# Shared-secret mà denso_backend gửi kèm header ``X-Internal-Token``.
# Rỗng = tắt auth (chỉ dành cho localhost). Production BẮT BUỘC đặt.
INTERNAL_API_TOKEN: str = _get_str("INTERNAL_API_TOKEN", "")

# ---------------------------------------------------------------------------
# Celery (tùy chọn — chỉ cần khi denso_backend dùng RAG_INGEST_MODE=celery)
# ---------------------------------------------------------------------------
CELERY_BROKER_URL: str = _get_str("CELERY_BROKER_URL", "redis://localhost:6379/0")
CELERY_RESULT_BACKEND: str = _get_str("CELERY_RESULT_BACKEND", "")
CELERY_TASK_QUEUE: str = _get_str("CELERY_TASK_QUEUE", "rag")

# ---------------------------------------------------------------------------
# HTTP API (dùng khi denso_backend gọi trực tiếp, không qua broker)
# ---------------------------------------------------------------------------
API_HOST: str = _get_str("API_HOST", "0.0.0.0")
API_PORT: int = _get_int("API_PORT", 8200)
LOG_LEVEL: str = _get_str("LOG_LEVEL", "INFO")
