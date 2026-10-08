"""
Entrypoint của microservice ``rag_worker``.

Hai giao diện, cùng gọi chung ``ingestion.ingest_document``:

1. **HTTP API** (FastAPI) — mặc định, không cần broker:

       uvicorn main:app --host 0.0.0.0 --port 8200

   - ``GET  /health``  : kiểm tra service còn sống.
   - ``GET  /ready``   : kiểm tra DB/schema/cấu hình embedding.
   - ``POST /ingest``  : nạp một tài liệu (``raw_data_id`` + ``file_url``).

2. **Celery worker** — dùng khi ``denso_backend`` đặt ``RAG_INGEST_MODE=celery``:

       celery -A main:celery_app worker -Q rag -l info --pool=solo

   Task: ``process.rag.task(raw_data_id, file_url)``.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

import settings
from broker_setup.config import celery_app
from database import controllers as db
from ingestion import ingest_document
from security import require_internal_token, warn_if_disabled

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

logging.getLogger("sentence_transformers").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Schema HTTP
# ---------------------------------------------------------------------------
class IngestRequest(BaseModel):
    """Yêu cầu nạp một tài liệu."""

    raw_data_id: str = Field(..., min_length=1, description="UUID của RawData.")
    file_url: str = Field(
        ...,
        min_length=1,
        description="URL (http/https) hoặc đường dẫn file cần nạp.",
    )
    title: Optional[str] = Field(default=None, description="Tiêu đề (chỉ để log).")


class IngestResponse(BaseModel):
    """Kết quả nạp một tài liệu."""

    raw_data_id: str
    status: str
    chunks: int
    characters: int
    elapsed_ms: int


class ReadyResponse(BaseModel):
    """Tình trạng sẵn sàng phục vụ."""

    status: str
    database_configured: bool
    schema_ready: bool
    missing_schema: list
    embedding_model: str
    embedding_dim: int
    vector_column_dim: Optional[int] = None
    dimension_ok: bool
    celery_broker: str
    auth_required: bool


# ---------------------------------------------------------------------------
# HTTP API
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(_: FastAPI):
    """Log cấu hình lúc khởi động."""
    logger.info("rag_worker khởi động")
    logger.info(
        "Embedding: model=%s dim=%d strict=%s",
        settings.EMBEDDING_MODEL,
        settings.EMBEDDING_DIM,
        settings.STRICT_EMBEDDING_DIM,
    )
    logger.info("Vector DB configured: %s", bool(settings.DATABASE_URL))
    logger.info("Celery broker: %s", settings.CELERY_BROKER_URL)
    warn_if_disabled("rag_worker")
    yield


app = FastAPI(
    title="DENSO RAG Worker",
    version="1.0.0",
    description=(
        "Microservice nạp tài liệu vào vector DB (pgvector): bóc tách nội dung, "
        "chia chunk, tạo embedding và ghi vào bảng rag_chunkdata."
    ),
    lifespan=lifespan,
)


@app.get("/health", tags=["system"])
async def health() -> dict:
    """Kiểm tra service còn sống (không truy cập DB/model)."""
    return {"status": "ok"}


@app.get("/ready", response_model=ReadyResponse, tags=["system"])
async def ready() -> ReadyResponse:
    """Kiểm tra DB, schema và cấu hình số chiều embedding.

    Endpoint này giúp phát hiện sớm hai lỗi hay gặp: chưa chạy migration, và
    ``EMBEDDING_DIM`` lệch với cột ``vector`` trong DB.
    """
    missing: list = []
    vector_dim: Optional[int] = None
    if settings.DATABASE_URL:
        try:
            missing = db.ensure_schema_ready()
            if "rag_chunkdata" not in missing:
                vector_dim = db.get_vector_column_dimensions()
        except Exception as exc:  # noqa: BLE001 - báo trạng thái, không crash
            logger.warning("Không kiểm tra được schema: %s", exc)
            missing = [f"error: {type(exc).__name__}: {exc}"]

    dimension_ok = bool(vector_dim) and vector_dim == settings.EMBEDDING_DIM
    return ReadyResponse(
        status="ready" if not missing and dimension_ok else "degraded",
        database_configured=bool(settings.DATABASE_URL),
        schema_ready=not missing,
        missing_schema=missing,
        embedding_model=settings.EMBEDDING_MODEL,
        embedding_dim=settings.EMBEDDING_DIM,
        vector_column_dim=vector_dim,
        dimension_ok=dimension_ok,
        celery_broker=settings.CELERY_BROKER_URL,
        auth_required=bool(settings.INTERNAL_API_TOKEN),
    )


@app.post(
    "/ingest",
    response_model=IngestResponse,
    tags=["ingest"],
    dependencies=[Depends(require_internal_token)],
)
def ingest(request: IngestRequest) -> IngestResponse:
    """Nạp một tài liệu vào vector DB (chạy đồng bộ).

    Là ``def`` (không phải ``async def``) nên FastAPI tự chạy trong threadpool,
    không chặn event loop khi embedding/tải file tốn thời gian.

    Raises:
        HTTPException 500: khi nạp thất bại (chi tiết nằm trong ``detail``).
    """
    try:
        result = ingest_document(request.raw_data_id, request.file_url)
    except Exception as exc:  # noqa: BLE001 - trả lỗi rõ ràng cho client
        raise HTTPException(
            status_code=500,
            detail=f"Nạp dữ liệu thất bại: {type(exc).__name__}: {exc}",
        ) from exc
    return IngestResponse(**result)


# ---------------------------------------------------------------------------
# Celery task (dùng khi RAG_INGEST_MODE=celery ở denso_backend)
# ---------------------------------------------------------------------------
@celery_app.task(name="process.rag.task", ignore_result=True)
def process_rag_task(raw_data_id: str, file_url: str) -> Optional[dict]:
    """Task Celery: nạp một tài liệu vào vector DB.

    Trạng thái được ghi vào ``rag_rawdata.status`` (``done``/``failed``), nên
    ``denso_backend`` không cần đọc kết quả trả về.
    """
    return ingest_document(raw_data_id, file_url)
