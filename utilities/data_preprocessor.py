"""Bóc tách nội dung và chia nhỏ văn bản thành chunk."""

from __future__ import annotations

import logging
import re
import uuid
from typing import Dict, List, Optional

import settings
from utilities.files_handler import open_file

logger = logging.getLogger(__name__)

#: Các đoạn văn được phân tách bằng dòng trống.
_PARAGRAPH_RE = re.compile(r"\n\s*\n")


def content_extractor(file_path: str) -> str:
    """Đọc và trả về nội dung text của file."""
    return open_file(file_path)


def _hard_split(paragraph: str, chunk_size: int) -> List[str]:
    """Cắt một đoạn văn dài hơn ``chunk_size`` thành nhiều mảnh.

    Ưu tiên cắt ở cuối câu (``. ``), sau đó tới dòng mới, rồi tới dấu cách;
    cuối cùng mới cắt cứng theo số ký tự.
    """
    if len(paragraph) <= chunk_size:
        return [paragraph]

    parts: List[str] = []
    remaining = paragraph
    while len(remaining) > chunk_size:
        window = remaining[:chunk_size]
        candidates = [window.rfind(". "), window.rfind("\n"), window.rfind(" ")]
        cut = max(candidates)
        if cut < chunk_size // 2:
            cut = chunk_size - 1
        parts.append(remaining[: cut + 1].strip())
        remaining = remaining[cut + 1 :].strip()

    if remaining:
        parts.append(remaining)
    return [part for part in parts if part]


def split_text(
    text: str,
    chunk_size: Optional[int] = None,
    min_chunk_size: Optional[int] = None,
    source: Optional[str] = None,
) -> List[Dict]:
    """Chia văn bản thành các chunk theo ranh giới đoạn văn.

    Quy tắc:
    1. Tách theo dòng trống thành các đoạn văn.
    2. Gom các đoạn liền kề cho tới khi đạt ``chunk_size`` ký tự.
    3. Đoạn nào dài hơn ``chunk_size`` bị cắt nhỏ (ưu tiên ranh giới câu).
    4. Chunk nhỏ hơn ``min_chunk_size`` được gộp vào chunk liền trước để tránh
       các chunk rác một hai dòng.

    Returns:
        Danh sách chunk dạng ``{"id", "text", "metadata"}``.
    """
    chunk_size = chunk_size or settings.CHUNK_SIZE
    min_chunk_size = min_chunk_size if min_chunk_size is not None else settings.MIN_CHUNK_SIZE

    normalized = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []

    paragraphs = [p.strip() for p in _PARAGRAPH_RE.split(normalized) if p.strip()]

    texts: List[str] = []
    current = ""
    for paragraph in paragraphs:
        for piece in _hard_split(paragraph, chunk_size):
            if not current:
                current = piece
            elif len(current) + 2 + len(piece) <= chunk_size:
                current = f"{current}\n\n{piece}"
            else:
                texts.append(current)
                current = piece
    if current:
        texts.append(current)

    # Gộp chunk quá nhỏ vào chunk liền trước (giữ nguyên nội dung, không mất chữ).
    merged: List[str] = []
    for chunk in texts:
        if merged and len(chunk) < min_chunk_size:
            merged[-1] = f"{merged[-1]}\n\n{chunk}"
        else:
            merged.append(chunk)

    return [
        {
            "id": str(uuid.uuid4()),
            "text": chunk,
            "metadata": {
                "source": source,
                "chunk_index": index,
                "char_count": len(chunk),
            },
        }
        for index, chunk in enumerate(merged)
    ]


def split_chunk_data(
    file_path: str,
    chunk_size: Optional[int] = None,
) -> List[Dict]:
    """Đọc file rồi chia nội dung thành chunk (giữ tương thích hàm cũ)."""
    content = content_extractor(file_path)
    return split_text(content, chunk_size=chunk_size, source=file_path)
