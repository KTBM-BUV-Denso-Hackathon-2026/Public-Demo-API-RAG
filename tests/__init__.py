"""Package chứa test của rag_worker.

Test chạy offline và không phụ thuộc ``.env`` thật, nên ta tắt auth nội bộ mặc
định cho cả bộ test. ``tests/test_pipeline.py::SecurityTests`` tự bật lại bằng
``mock.patch`` để kiểm tra hành vi bật/tắt.
"""

import settings

# Rỗng = auth tắt -> POST /ingest không đòi header X-Internal-Token.
settings.INTERNAL_API_TOKEN = ""
