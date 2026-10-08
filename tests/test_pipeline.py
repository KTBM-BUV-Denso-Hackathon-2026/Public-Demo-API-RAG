"""Test cho ``rag_worker`` (chạy offline, không cần DB/model thật).

Các tầng DB và embedding được mock để test nhanh và độc lập. Bản kiểm chứng
với PostgreSQL + model thật thực hiện qua ``POST /ingest`` (xem README).
"""

import logging
import os
import re
import sys
import unittest
from unittest import mock

# Giảm log nhiễu từ các test cố tình gây lỗi.
logging.disable(logging.CRITICAL)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import settings  # noqa: E402
from database import controllers as db  # noqa: E402
from embedding_engine import config as embedding_config  # noqa: E402
from tests import fixtures  # noqa: E402
from utilities.data_preprocessor import split_text  # noqa: E402
from utilities.files_handler import resolve_local_path  # noqa: E402

RAW_ID = "8b0ae028-35bf-487f-9bf0-03eeb1942aa4"


class ChunkingTests(unittest.TestCase):
    def test_empty_text_returns_no_chunks(self):
        self.assertEqual(split_text("   \n\n  "), [])

    def test_paragraphs_are_packed_under_chunk_size(self):
        paragraph = "A" * 400
        text = "\n\n".join([paragraph] * 4)

        chunks = split_text(text, chunk_size=1000, min_chunk_size=80)

        self.assertTrue(chunks)
        for chunk in chunks:
            self.assertLessEqual(len(chunk["text"]), 1000)
        # Không được mất chữ nào: bỏ qua khác biệt về dấu phân cách đoạn.
        self.assertEqual(
            re.sub(r"\s+", "", "".join(chunk["text"] for chunk in chunks)),
            re.sub(r"\s+", "", text),
        )

    def test_chunk_size_is_respected_for_single_huge_paragraph(self):
        text = "câu. " * 500  # ~2500 ký tự, không có dòng trống

        chunks = split_text(text, chunk_size=500, min_chunk_size=10)

        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk["text"]), 500)

    def test_short_tail_is_merged_into_previous_chunk(self):
        text = ("B" * 900) + "\n\n" + "ngắn"

        chunks = split_text(text, chunk_size=1000, min_chunk_size=80)

        self.assertEqual(len(chunks), 1)
        self.assertIn("ngắn", chunks[0]["text"])

    def test_metadata_and_ids_are_set(self):
        chunks = split_text("Nội dung thử nghiệm", chunk_size=1000, source="a.md")

        self.assertEqual(len(chunks), 1)
        chunk = chunks[0]
        self.assertTrue(chunk["id"])
        self.assertEqual(chunk["metadata"]["source"], "a.md")
        self.assertEqual(chunk["metadata"]["chunk_index"], 0)
        self.assertEqual(chunk["metadata"]["char_count"], len(chunk["text"]))


class FilePathTests(unittest.TestCase):
    def test_media_url_is_joined_with_media_root(self):
        media_root = os.path.join(os.sep, "tmp", "media-root")

        with mock.patch.object(settings, "MEDIA_ROOT", media_root):
            path = resolve_local_path("/media/rag_files/a.md")

        self.assertTrue(path.replace("\\", "/").endswith("media-root/rag_files/a.md"))

    def test_absolute_path_is_left_alone(self):
        absolute = os.path.join(os.sep, "tmp", "khong-ton-tai.md")

        self.assertEqual(resolve_local_path(absolute), absolute)

    def test_file_scheme_is_stripped(self):
        self.assertTrue(resolve_local_path("file:///tmp/a.md").endswith("a.md"))


class VectorWriteTests(unittest.TestCase):
    def test_vector_is_formatted_for_pgvector(self):
        self.assertEqual(db.to_vector_literal([0.5, -1, 2.0]), "[0.5,-1.0,2.0]")

    def test_replace_chunks_rejects_length_mismatch(self):
        with self.assertRaises(ValueError):
            db.replace_chunks(RAW_ID, [{"text": "a"}], [])

    def test_replace_chunks_deletes_then_inserts(self):
        cursor = fixtures.RecordingCursor()
        chunks = [{"text": "nội dung", "metadata": {"source": "a.md"}}]

        with fixtures.patch_db(cursor), mock.patch.object(db, "execute_values") as insert:
            written = db.replace_chunks(RAW_ID, chunks, [[0.1, 0.2]])

        self.assertEqual(written, 1)
        delete_sql, delete_params = cursor.executed[0]
        self.assertIn("DELETE FROM rag_chunkdata", delete_sql)
        self.assertEqual(delete_params, (RAW_ID,))

        sql = insert.call_args.args[1]
        rows = insert.call_args.args[2]
        template = insert.call_args.kwargs["template"]
        self.assertIn("INSERT INTO rag_chunkdata", sql)
        for column in ("id", "raw_data_id", "chunk_content", "vector", "fts"):
            self.assertIn(column, sql)
        self.assertIn("::vector", template)
        self.assertIn("to_tsvector", template)
        self.assertEqual(rows[0][2], "nội dung")  # chunk_content
        self.assertEqual(rows[0][5], "[0.1,0.2]")  # literal pgvector

    def test_set_status_writes_expected_values(self):
        cursor = fixtures.RecordingCursor()

        with fixtures.patch_db(cursor):
            db.set_raw_data_status(RAW_ID, "failed", "lỗi gì đó")

        sql, params = cursor.executed[0]
        self.assertIn("UPDATE rag_rawdata", sql)
        self.assertEqual(params, ("failed", "lỗi gì đó", RAW_ID))


class DimensionGuardTests(unittest.TestCase):
    def test_validate_dimension_raises_when_strict(self):
        with mock.patch.object(settings, "EMBEDDING_DIM", 768), mock.patch.object(
            settings, "STRICT_EMBEDDING_DIM", True
        ):
            with self.assertRaises(RuntimeError):
                embedding_config.validate_dimension(384)

    def test_validate_dimension_warns_when_not_strict(self):
        with mock.patch.object(settings, "EMBEDDING_DIM", 768), mock.patch.object(
            settings, "STRICT_EMBEDDING_DIM", False
        ), mock.patch.object(embedding_config.logger, "warning") as warn:
            embedding_config.validate_dimension(384)

        warn.assert_called_once()

    def test_matching_dimension_is_silent(self):
        with mock.patch.object(settings, "EMBEDDING_DIM", 768), mock.patch.object(
            embedding_config.logger, "warning"
        ) as warn:
            embedding_config.validate_dimension(768)

        warn.assert_not_called()


class IngestionPipelineTests(unittest.TestCase):
    """Kiểm tra trình tự và trạng thái của ``ingest_document``."""

    def test_success_path_sets_processing_then_done(self):
        import ingestion

        vectors = [[0.1] * 768]
        with mock.patch.object(db, "raw_data_exists", return_value=True), mock.patch.object(
            db, "set_raw_data_status"
        ) as status, mock.patch.object(
            db, "update_raw_data_content"
        ) as content, mock.patch.object(
            db, "replace_chunks", return_value=1
        ) as chunks, mock.patch.object(
            ingestion, "fetch_document", return_value=("f.md", False)
        ), mock.patch.object(
            ingestion, "content_extractor", return_value="Nội dung tài liệu."
        ), mock.patch.object(
            ingestion, "chunks_process", return_value=vectors
        ):
            result = ingestion.ingest_document(RAW_ID, "http://x/f.md")

        self.assertEqual(result["status"], "done")
        self.assertEqual(result["chunks"], 1)
        content.assert_called_once()
        chunks.assert_called_once()
        self.assertEqual(
            [call.args[1] for call in status.call_args_list], ["processing", "done"]
        )

    def test_missing_raw_data_marks_failed(self):
        import ingestion

        with mock.patch.object(db, "raw_data_exists", return_value=False), mock.patch.object(
            db, "set_raw_data_status"
        ) as status:
            with self.assertRaises(ValueError):
                ingestion.ingest_document(RAW_ID, "http://x/f.md")

        self.assertEqual(status.call_args.args[1], "failed")

    def test_empty_document_marks_failed(self):
        import ingestion

        with mock.patch.object(db, "raw_data_exists", return_value=True), mock.patch.object(
            db, "set_raw_data_status"
        ) as status, mock.patch.object(
            ingestion, "fetch_document", return_value=("f.md", False)
        ), mock.patch.object(
            ingestion, "content_extractor", return_value="   "
        ):
            with self.assertRaises(ValueError):
                ingestion.ingest_document(RAW_ID, "http://x/f.md")

        self.assertEqual(status.call_args.args[1], "failed")

    def test_downloaded_file_is_deleted(self):
        import ingestion

        with mock.patch.object(db, "raw_data_exists", return_value=True), mock.patch.object(
            db, "set_raw_data_status"
        ), mock.patch.object(db, "update_raw_data_content"), mock.patch.object(
            db, "replace_chunks", return_value=1
        ), mock.patch.object(
            ingestion, "fetch_document", return_value=("tmp.md", True)
        ), mock.patch.object(
            ingestion, "content_extractor", return_value="x" * 200
        ), mock.patch.object(
            ingestion, "chunks_process", return_value=[[0.1]]
        ), mock.patch.object(
            ingestion, "delete_file"
        ) as delete:
            ingestion.ingest_document(RAW_ID, "http://x/f.md")

        delete.assert_called_once_with("tmp.md")

    def test_local_file_is_kept(self):
        import ingestion

        with mock.patch.object(db, "raw_data_exists", return_value=True), mock.patch.object(
            db, "set_raw_data_status"
        ), mock.patch.object(db, "update_raw_data_content"), mock.patch.object(
            db, "replace_chunks", return_value=1
        ), mock.patch.object(
            ingestion, "fetch_document", return_value=("local.md", False)
        ), mock.patch.object(
            ingestion, "content_extractor", return_value="x" * 200
        ), mock.patch.object(
            ingestion, "chunks_process", return_value=[[0.1]]
        ), mock.patch.object(
            ingestion, "delete_file"
        ) as delete:
            ingestion.ingest_document(RAW_ID, "local.md")

        delete.assert_not_called()


class DimensionPrecheckTests(unittest.TestCase):
    """``_assert_dimension_matches_db`` phải chặn lỗi số chiều TRƯỚC khi làm việc."""

    def test_matching_dimension_passes(self):
        import ingestion

        with mock.patch.object(db, "get_vector_column_dimensions", return_value=768):
            with mock.patch.object(settings, "EMBEDDING_DIM", 768):
                ingestion._assert_dimension_matches_db()  # không ném

    def test_mismatch_raises_with_actionable_message(self):
        import ingestion

        with mock.patch.object(db, "get_vector_column_dimensions", return_value=384):
            with mock.patch.object(settings, "EMBEDDING_DIM", 768):
                with self.assertRaises(RuntimeError) as ctx:
                    ingestion._assert_dimension_matches_db()

        message = str(ctx.exception)
        self.assertIn("384", message)
        self.assertIn("768", message)
        self.assertIn("EMBEDDING_DIM", message)

    def test_unknown_column_dimension_is_skipped(self):
        """Chưa migrate (đọc không được) thì bỏ qua, để ``/ready`` báo."""
        import ingestion

        with mock.patch.object(db, "get_vector_column_dimensions", return_value=None):
            with mock.patch.object(ingestion.logger, "warning") as warn:
                ingestion._assert_dimension_matches_db()

        warn.assert_called()

    def test_dimension_is_checked_before_fetching_the_file(self):
        """Thứ tự quan trọng: lỗi cấu hình phải nổ trước khi tốn công tải/embedding."""
        import ingestion

        fetched = []
        statuses = []
        with mock.patch.object(db, "raw_data_exists", return_value=True):
            with mock.patch.object(db, "get_vector_column_dimensions", return_value=384):
                with mock.patch.object(settings, "EMBEDDING_DIM", 768):
                    with mock.patch.object(
                        db,
                        "set_raw_data_status",
                        side_effect=lambda rid, st, err="": statuses.append(st),
                    ):
                        with mock.patch.object(
                            ingestion, "fetch_document", side_effect=lambda *a, **k: fetched.append(1)
                        ):
                            with self.assertRaises(RuntimeError):
                                ingestion.ingest_document(RAW_ID, "http://x/f.md")

        # Không tải file, không embedding -> tiết kiệm toàn bộ công việc nặng.
        self.assertEqual(fetched, [])
        # Vẫn ghi failed để client đọc được nguyên nhân qua API.
        self.assertEqual(statuses, ["failed"])


class SecurityTests(unittest.TestCase):
    """Auth nội bộ: bật khi có token, tắt khi rỗng, từ chối sai token."""

    def test_header_name(self):
        import security

        self.assertEqual(security.HEADER_NAME, "X-Internal-Token")

    def test_disabled_when_token_empty(self):
        import asyncio
        import security

        with mock.patch.object(settings, "INTERNAL_API_TOKEN", ""):
            self.assertFalse(security.is_enabled())
            # Không token -> vẫn cho qua (chế độ dev).
            asyncio.run(security.require_internal_token(x_internal_token=None))

    def test_correct_token_passes(self):
        import asyncio
        import security

        with mock.patch.object(settings, "INTERNAL_API_TOKEN", "abc123"):
            self.assertTrue(security.is_enabled())
            asyncio.run(security.require_internal_token(x_internal_token="abc123"))

    def test_missing_token_rejected(self):
        import asyncio
        import security

        with mock.patch.object(settings, "INTERNAL_API_TOKEN", "abc123"):
            with self.assertRaises(Exception) as ctx:
                asyncio.run(security.require_internal_token(x_internal_token=None))
        self.assertEqual(getattr(ctx.exception, "status_code", None), 401)

    def test_wrong_token_rejected(self):
        import asyncio
        import security

        with mock.patch.object(settings, "INTERNAL_API_TOKEN", "abc123"):
            for bad in ("", "abc124", "abc1234", "ABC123"):
                with self.assertRaises(Exception):
                    asyncio.run(security.require_internal_token(x_internal_token=bad))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
