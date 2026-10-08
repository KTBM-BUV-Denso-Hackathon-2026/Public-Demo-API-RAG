"""Chuyển chunk văn bản thành vector embedding."""

from __future__ import annotations

import logging
from typing import Dict, List

from embedding_engine.config import encode_texts, get_embedding

logger = logging.getLogger(__name__)


def chunk_to_vector(chunk: str) -> List[float]:
    """Chuyển một chunk văn bản thành vector embedding."""
    return get_embedding(chunk)


def chunks_process(chunks: List[Dict]) -> List[List[float]]:
    """Vector hoá toàn bộ chunk trong một lần encode.

    Args:
        chunks: Danh sách chunk, mỗi phần tử có khoá ``text``.

    Returns:
        Danh sách vector, giữ đúng thứ tự với ``chunks``.
    """
    if not chunks:
        return []

    texts = [chunk["text"] for chunk in chunks]
    vectors = encode_texts(texts)
    logger.info("Đã vector hoá %d chunk", len(vectors))
    return vectors
