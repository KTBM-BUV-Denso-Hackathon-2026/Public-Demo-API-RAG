"""Tải / đọc / xoá file cho quá trình nạp dữ liệu."""

from __future__ import annotations

import logging
import os
import tempfile
import uuid
from typing import Optional, Tuple
from urllib.parse import unquote, urlparse

import requests

import settings

logger = logging.getLogger(__name__)

_CHUNK_BYTES = 1024 * 64


def resolve_local_path(file_url: str) -> str:
    """Chuyển URL/đường dẫn thành đường dẫn file trên máy này.

    Hỗ trợ:
    - ``file:///abs/path``            → bỏ scheme.
    - ``/media/rag_files/x.md``       → ghép ``MEDIA_ROOT`` (bỏ tiền tố MEDIA_URL).
    - ``rag_files/x.md``              → ghép ``MEDIA_ROOT``.
    - ``/abs/path`` hoặc ``C:\\x.md`` → dùng nguyên (nếu file tồn tại).
    """
    if file_url.startswith("file://"):
        parsed = urlparse(file_url)
        path = unquote(parsed.path)
        # Bỏ dấu '/' đầu trên Windows (ví dụ '/C:/x' → 'C:/x').
        if os.name == "nt" and len(path) > 2 and path[0] == "/" and path[2] == ":":
            path = path[1:]
        return path

    media_url = settings.MEDIA_URL or "/media/"
    if settings.MEDIA_ROOT:
        relative = None
        if file_url.startswith(media_url):
            relative = file_url[len(media_url) :]
        elif not os.path.isabs(file_url):
            relative = file_url
        if relative is not None:
            candidate = os.path.join(settings.MEDIA_ROOT, relative.lstrip("/\\"))
            if os.path.isfile(candidate) or not os.path.isfile(file_url):
                return candidate

    return file_url


def download_file(
    url: str,
    destination: Optional[str] = None,
    timeout: Optional[float] = None,
) -> str:
    """Tải file từ URL về đĩa, trả về đường dẫn đã lưu.

    Raises:
        RuntimeError: nếu HTTP trả về mã lỗi.
    """
    if destination is None:
        suffix = os.path.splitext(urlparse(url).path)[1] or ".txt"
        destination = os.path.join(
            tempfile.gettempdir(), f"rag_{uuid.uuid4().hex}{suffix}"
        )

    timeout = timeout or settings.DOWNLOAD_TIMEOUT
    try:
        response = requests.get(url, stream=True, timeout=timeout)
    except requests.RequestException as exc:
        raise RuntimeError(f"Không tải được file từ {url}: {exc}") from exc

    with response:
        if response.status_code != 200:
            raise RuntimeError(
                f"Tải file thất bại: HTTP {response.status_code} từ {url}"
            )
        with open(destination, "wb") as handle:
            for chunk in response.iter_content(_CHUNK_BYTES):
                if chunk:
                    handle.write(chunk)

    logger.info("Đã tải file %s → %s", url, destination)
    return destination


def fetch_document(file_url: str) -> Tuple[str, bool]:
    """Lấy tài liệu về máy để xử lý.

    Returns:
        ``(đường_dẫn, cần_xoá_sau_khi_dùng)`` — file local không bị xoá.
    """
    if file_url.startswith(("http://", "https://")):
        return download_file(file_url), True

    path = resolve_local_path(file_url)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Không tìm thấy file: {path}")
    return path, False


def open_file(file_path: str) -> str:
    """Đọc nội dung file text (UTF-8; ký tự lỗi được thay bằng U+FFFD)."""
    with open(file_path, "r", encoding="utf-8", errors="replace") as handle:
        return handle.read()


def delete_file(file_path: str) -> bool:
    """Xoá file; trả về True nếu xoá thành công."""
    try:
        os.remove(file_path)
        return True
    except OSError as exc:
        logger.warning("Không xoá được file %s: %s", file_path, exc)
        return False
