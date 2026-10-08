"""Xác thực nội bộ cho ``rag_worker``.

Bản sao có chủ đích của ``llm_api/security.py``: hai service được thiết kế để
deploy độc lập (mỗi cái có ``.env``/``requirements.txt`` riêng) nên không import
chéo. Nếu sau này gom chung, hãy tách module này ra package dùng chung.

Vì sao cần: ``POST /ingest`` gọi ``replace_chunks`` — xoá sạch chunk cũ rồi ghi
lại. Không có lớp bảo vệ thì bất kỳ ai biết cổng 8200 cũng ghi đè được dữ liệu
tri thức của hệ thống.
"""

from __future__ import annotations

import hmac
import logging
from typing import Optional

from fastapi import Header, HTTPException, status

import settings

logger = logging.getLogger(__name__)

#: Tên header mà ``denso_backend`` gửi kèm mọi request nội bộ.
HEADER_NAME = "X-Internal-Token"

_UNAUTHORIZED = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail=f"Thiếu hoặc sai header {HEADER_NAME} (kiểm tra INTERNAL_API_TOKEN).",
)

_warned_disabled = False


def is_enabled() -> bool:
    """True khi đã cấu hình ``INTERNAL_API_TOKEN`` (tức auth đang bật)."""
    return bool(settings.INTERNAL_API_TOKEN)


def warn_if_disabled(service: str) -> None:
    """Cảnh báo một lần nếu đang chạy mà chưa bật auth."""
    global _warned_disabled
    if not is_enabled() and not _warned_disabled:
        _warned_disabled = True
        logger.warning(
            "⚠ %s đang chạy KHÔNG có xác thực nội bộ (INTERNAL_API_TOKEN rỗng). "
            "Chỉ nên dùng ở localhost; production bắt buộc đặt INTERNAL_API_TOKEN.",
            service,
        )


async def require_internal_token(
    x_internal_token: Optional[str] = Header(default=None, alias=HEADER_NAME),
) -> None:
    """FastAPI dependency: chặn request không hợp lệ.

    Dùng ``Optional[str]`` chứ không phải ``str | None`` vì FastAPI phải *đánh giá
    thật* annotation này để đọc tên header, mà toán tử ``|`` cho union chỉ có từ
    Python 3.10 (workspace đang dùng 3.9).

    Raises:
        HTTPException 401: khi auth đang bật mà header thiếu hoặc sai.
    """
    expected = settings.INTERNAL_API_TOKEN
    if not expected:
        return  # Auth tắt -> cho qua (chế độ dev).
    if not x_internal_token or not hmac.compare_digest(x_internal_token, expected):
        raise _UNAUTHORIZED
