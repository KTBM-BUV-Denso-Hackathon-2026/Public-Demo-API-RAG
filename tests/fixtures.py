"""Tiện ích giả lập DB cho test (không cần PostgreSQL)."""

from contextlib import contextmanager
from unittest import mock

from database import controllers


class RecordingCursor:
    """Cursor giả: ghi lại câu lệnh đã thực thi."""

    def __init__(self):
        self.executed = []

    def execute(self, query, params=None):
        self.executed.append((query, params))

    def fetchone(self):
        return (1,)

    def fetchall(self):
        return []

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


class FakeConnection:
    def __init__(self, cursor):
        self._cursor = cursor
        self.committed = False

    def cursor(self):
        return self._cursor

    def commit(self):
        self.committed = True

    def rollback(self):  # pragma: no cover - chỉ dùng khi lỗi
        pass


def patch_db(cursor):
    """Trả về context manager thay ``database.controllers.get_db`` bằng DB giả."""

    @contextmanager
    def _fake_get_db():
        yield FakeConnection(cursor)

    return mock.patch.object(controllers, "get_db", _fake_get_db)
