import pytest

from app.api import upload
from app.api.upload import _is_pdf_filename
from app.services.progress import create_task, get_progress


@pytest.mark.parametrize("filename", ["report.pdf", "report.PDF", "report.PdF"])
def test_pdf_extension_is_case_insensitive(filename):
    assert _is_pdf_filename(filename)

@pytest.mark.parametrize("filename", [None, "", "report.pdf.bak", "report.txt"])
def test_non_pdf_filename_is_rejected(filename):
    assert not _is_pdf_filename(filename)


def test_skipped_ingestion_is_reported_as_error(monkeypatch, tmp_path):
    class AlreadyIngested:
        def run_pipeline(self, **_kwargs):
            return {"status": "skipped"}

    monkeypatch.setattr(upload, "DocumentIngestionService", AlreadyIngested)
    pdf_path = tmp_path / "report.pdf"
    pdf_path.write_bytes(b"pdf-content")
    task_id = "upload-test"
    create_task(task_id, "report.pdf")

    upload.process_and_ingest_document(str(pdf_path), "report.pdf", task_id)

    progress = get_progress(task_id)
    assert progress["status"] == "error"
    assert "已存在" in progress["error"]
