"""
Nạp model embedding (sentence-transformers) và kiểm tra số chiều.

Model được nạp **lười** (lazy) và cache lại: ``import`` module này không tải
model, nhờ đó API/Celery khởi động nhanh và không cần model khi chỉ health-check.
"""

from __future__ import annotations

import logging
import threading
from typing import Dict, List, Optional

import settings

logger = logging.getLogger(__name__)

# Cache model theo tên: {model_name: SentenceTransformer}
_MODELS: Dict[str, object] = {}
_LOCK = threading.Lock()


def get_model(model_name: Optional[str] = None):
    """Trả về ``SentenceTransformer`` đã cache cho ``model_name``.

    Model chỉ được nạp một lần cho mỗi tên; an toàn với đa luồng
    (double-checked locking).
    """
    name = model_name or settings.EMBEDDING_MODEL
    model = _MODELS.get(name)
    if model is not None:
        return model

    with _LOCK:
        model = _MODELS.get(name)
        if model is None:
            # Import muộn: torch/sentence-transformers rất nặng.
            from sentence_transformers import SentenceTransformer

            logger.info("Đang nạp embedding model: %s", name)
            model = SentenceTransformer(name)
            _MODELS[name] = model
            dimension = get_model_dimension(name)
            logger.info("Đã nạp model %s (dim=%s)", name, dimension)
            validate_dimension(dimension)
    return model


def get_model_dimension(model_name: Optional[str] = None) -> Optional[int]:
    """Số chiều vector của model, hoặc ``None`` nếu model không hỗ trợ."""
    model = get_model(model_name)
    try:
        return int(model.get_sentence_embedding_dimension())
    except Exception:  # pragma: no cover - phụ thuộc phiên bản thư viện
        return None


def validate_dimension(dimension: Optional[int]) -> None:
    """Kiểm tra số chiều model có khớp ``EMBEDDING_DIM`` không.

    Đây là chốt chặn quan trọng: nếu model trả về số chiều khác cột
    ``vector`` trong DB, mọi kết quả tìm kiếm sẽ sai một cách âm thầm.
    """
    expected = settings.EMBEDDING_DIM
    if not expected or dimension is None or dimension == expected:
        return

    message = (
        f"Sai số chiều embedding: model '{settings.EMBEDDING_MODEL}' trả về "
        f"{dimension} chiều nhưng EMBEDDING_DIM={expected}. "
        f"Hãy sửa EMBEDDING_DIM (và cột vector trong DB) cho khớp."
    )
    if settings.STRICT_EMBEDDING_DIM:
        raise RuntimeError(message)
    logger.warning(message)


def get_embedding(text: str) -> List[float]:
    """Vector hoá một đoạn văn bản, trả về list[float]."""
    vectors = encode_texts([text])
    return vectors[0] if vectors else []


def encode_texts(
    texts: List[str],
    model_name: Optional[str] = None,
    batch_size: Optional[int] = None,
) -> List[List[float]]:
    """Vector hoá nhiều văn bản trong MỘT lần gọi (batch encode).

    ``normalize_embeddings=True`` để dùng cosine distance ổn định.
    """
    if not texts:
        return []

    model = get_model(model_name)
    embeddings = model.encode(
        texts,
        batch_size=batch_size or settings.EMBEDDING_BATCH_SIZE,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    vectors = [vector.tolist() for vector in embeddings]
    if vectors:
        validate_dimension(len(vectors[0]))
    return vectors


def reset_cache() -> None:
    """Xoá cache model (chủ yếu dùng cho test)."""
    with _LOCK:
        _MODELS.clear()
