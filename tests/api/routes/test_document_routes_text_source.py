"""Preserve full ``file_source`` for ``/documents/text(s)``.

Upload/scan still use a disk basename. List/track/paginated must return the
stored URI, and deleting a URL-backed document must not unlink ``INPUT/``.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_dr = importlib.import_module("lightrag.api.routers.document_routes")
_base = importlib.import_module("lightrag.base")
_utils_pipeline = importlib.import_module("lightrag.utils_pipeline")
sys.argv = _original_argv

InsertTextRequest = _dr.InsertTextRequest
create_document_routes = _dr.create_document_routes
delete_file_variants_by_file_path = _dr.delete_file_variants_by_file_path
get_existing_doc_by_file_path_candidates = _dr.get_existing_doc_by_file_path_candidates
pipeline_enqueue_file = _dr.pipeline_enqueue_file
DocProcessingStatus = _base.DocProcessingStatus
DocStatus = _base.DocStatus
canonicalize_document_source = _utils_pipeline.canonicalize_document_source

pytestmark = pytest.mark.offline

_HEADERS = {"X-API-Key": "test-key"}
_URL_A = "https://gitlab.example.com/group/repo-a/-/blob/main/README.md"
_URL_B = "https://gitlab.example.com/group/repo-b/-/blob/main/README.md"
_S3_A = "s3://bucket/project-a/docs/index.md"
_S3_B = "s3://bucket/project-b/docs/index.md"


def _doc(file_path: str, *, status: DocStatus = DocStatus.PROCESSED, suffix: str = "x"):
    return DocProcessingStatus(
        content_summary=f"{suffix} summary",
        content_length=10,
        file_path=file_path,
        status=status,
        created_at="2024-01-01T00:00:00+00:00",
        updated_at="2024-01-01T00:00:00+00:00",
        track_id="track-text",
        metadata={},
    )


class _ExactMatchDocStatus:
    """Production-like lookup: exact ``file_path`` match, not Path.name."""

    def __init__(self, docs=None):
        self.docs = dict(docs or {})

    def _stored_path(self, doc):
        return (
            doc.get("file_path")
            if isinstance(doc, dict)
            else getattr(doc, "file_path", None)
        )

    async def get_doc_by_file_path(self, file_path):
        for doc in self.docs.values():
            if self._stored_path(doc) == file_path:
                return doc
        return None

    async def get_doc_by_file_basename(self, basename):
        for doc_id, doc in self.docs.items():
            if self._stored_path(doc) == basename:
                return doc_id, doc
        return None

    async def get_docs_paginated(
        self,
        status_filter=None,
        status_filters=None,
        page=1,
        page_size=50,
        sort_field="updated_at",
        sort_direction="desc",
    ):
        items = list(self.docs.items())
        return items[:page_size], len(items)

    async def get_all_status_counts(self):
        return {"processed": len(self.docs)}


class _TextRag:
    workspace = "text-source-test"
    addon_params: dict = {}

    def __init__(self, doc_status=None):
        self.doc_status = doc_status or _ExactMatchDocStatus()

    async def get_docs_by_status(self, status):
        if status != DocStatus.PROCESSED:
            return {}
        return {
            doc_id: doc
            for doc_id, doc in self.doc_status.docs.items()
            if getattr(doc, "status", None) == DocStatus.PROCESSED
        }

    async def aget_docs_by_track_id(self, track_id):
        return {
            doc_id: doc
            for doc_id, doc in self.doc_status.docs.items()
            if getattr(doc, "track_id", None) == track_id
        }


class _EnqueueRag:
    def __init__(self):
        self.enqueued = []
        self.errors = []
        self.addon_params = {}

    async def apipeline_enqueue_documents(self, input, **kwargs):
        self.enqueued.append({"input": input, **kwargs})
        return kwargs.get("track_id") or "track-xyz"

    async def apipeline_enqueue_error_documents(self, error_files, track_id=None):
        self.errors.append((error_files, track_id))


def _make_text_client(monkeypatch, rag):
    captured: dict = {}

    async def _spy(_rag, texts, file_sources=None, track_id=None, chunking=None):
        captured.setdefault("calls", []).append(
            {
                "texts": texts,
                "file_sources": file_sources,
                "track_id": track_id,
            }
        )

    async def _noop_reserve(_rag):
        return False

    async def _noop_release(_rag):
        return None

    monkeypatch.setattr(_dr, "pipeline_index_texts", _spy)
    monkeypatch.setattr(_dr, "_reserve_enqueue_slot", _noop_reserve)
    monkeypatch.setattr(_dr, "_release_enqueue_slot", _noop_release)

    app = FastAPI()
    app.include_router(
        create_document_routes(rag, SimpleNamespace(), api_key="test-key")
    )
    return TestClient(app), captured


def test_insert_text_request_keeps_gitlab_url():
    req = InsertTextRequest.model_validate(
        {"text": "alpha from repo A", "file_source": _URL_A}
    )
    assert req.file_source == _URL_A


def test_insert_text_accepts_distinct_urls_with_same_basename(monkeypatch):
    client, captured = _make_text_client(monkeypatch, _TextRag())

    first = client.post(
        "/documents/text",
        headers=_HEADERS,
        json={"text": "alpha from repo A", "file_source": _URL_A},
    )
    second = client.post(
        "/documents/text",
        headers=_HEADERS,
        json={"text": "beta from repo B", "file_source": _URL_B},
    )

    assert first.status_code == 200
    assert second.status_code == 200
    assert captured["calls"][0]["file_sources"] == [_URL_A]
    assert captured["calls"][1]["file_sources"] == [_URL_B]


def test_insert_text_accepts_distinct_s3_keys(monkeypatch):
    client, captured = _make_text_client(monkeypatch, _TextRag())

    first = client.post(
        "/documents/text",
        headers=_HEADERS,
        json={"text": "index A", "file_source": _S3_A},
    )
    second = client.post(
        "/documents/text",
        headers=_HEADERS,
        json={"text": "index B", "file_source": _S3_B},
    )

    assert first.status_code == 200
    assert second.status_code == 200
    assert captured["calls"][0]["file_sources"] == [_S3_A]
    assert captured["calls"][1]["file_sources"] == [_S3_B]


def test_insert_text_strips_hint_from_url_last_segment(monkeypatch):
    client, captured = _make_text_client(monkeypatch, _TextRag())

    resp = client.post(
        "/documents/text",
        headers=_HEADERS,
        json={"text": "alpha", "file_source": "https://x/notes.[native].md"},
    )

    assert resp.status_code == 200
    assert captured["calls"][0]["file_sources"] == ["https://x/notes.md"]


def test_insert_text_hinted_url_conflicts_with_stored_plain_url(monkeypatch):
    rag = _TextRag(
        _ExactMatchDocStatus(
            {
                "doc-a": {
                    "status": DocStatus.PROCESSED.value,
                    "file_path": "https://x/notes.md",
                }
            }
        )
    )
    client, captured = _make_text_client(monkeypatch, rag)

    resp = client.post(
        "/documents/text",
        headers=_HEADERS,
        json={"text": "changed", "file_source": "https://x/notes.[native].md"},
    )

    assert resp.status_code == 409
    assert "https://x/notes.md" in resp.json()["detail"]
    assert captured == {}


def test_insert_text_same_url_returns_409(monkeypatch):
    rag = _TextRag(
        _ExactMatchDocStatus(
            {
                "doc-a": {
                    "status": DocStatus.PROCESSED.value,
                    "file_path": _URL_A,
                }
            }
        )
    )
    client, captured = _make_text_client(monkeypatch, rag)

    resp = client.post(
        "/documents/text",
        headers=_HEADERS,
        json={"text": "changed", "file_source": _URL_A},
    )

    assert resp.status_code == 409
    assert _URL_A in resp.json()["detail"]
    assert captured == {}


def test_insert_texts_duplicate_sources_in_one_request_are_400(monkeypatch):
    client, captured = _make_text_client(monkeypatch, _TextRag())

    resp = client.post(
        "/documents/texts",
        headers=_HEADERS,
        json={
            "texts": ["alpha", "beta"],
            "file_sources": [_URL_A, _URL_A],
        },
    )

    assert resp.status_code == 400
    assert "unique" in resp.json()["detail"].lower()
    assert captured == {}


def test_paginated_and_track_status_return_full_url(monkeypatch):
    rag = _TextRag(_ExactMatchDocStatus({"doc-a": _doc(_URL_A)}))
    app = FastAPI()
    app.include_router(
        create_document_routes(rag, SimpleNamespace(), api_key="test-key")
    )
    client = TestClient(app)

    paginated = client.post(
        "/documents/paginated",
        headers=_HEADERS,
        json={
            "page": 1,
            "page_size": 10,
            "sort_field": "updated_at",
            "sort_direction": "desc",
        },
    )
    assert paginated.status_code == 200
    assert paginated.json()["documents"][0]["file_path"] == _URL_A

    listed = client.get("/documents", headers=_HEADERS)
    assert listed.status_code == 200
    listed_docs = [doc for docs in listed.json()["statuses"].values() for doc in docs]
    assert listed_docs[0]["file_path"] == _URL_A

    tracked = client.get("/documents/track_status/track-text", headers=_HEADERS)
    assert tracked.status_code == 200
    assert tracked.json()["documents"][0]["file_path"] == _URL_A


@pytest.mark.asyncio
async def test_upload_basename_lookup_ignores_url_with_same_filename():
    doc_status = _ExactMatchDocStatus(
        {
            "doc-url": {
                "status": DocStatus.PROCESSED.value,
                "file_path": _URL_A,
            }
        }
    )
    match = await get_existing_doc_by_file_path_candidates(
        doc_status, Path("/app/data/inputs/README.md")
    )
    assert match is None

    match_url = await get_existing_doc_by_file_path_candidates(doc_status, _URL_A)
    # Upload lookup canonicalizes to Path.name, so a GitLab URL is not found
    # by basename either — only stored ``README.md`` would 409 an upload.
    assert match_url is None

    doc_status.docs["doc-local"] = {
        "status": DocStatus.PROCESSED.value,
        "file_path": "README.md",
    }
    local = await get_existing_doc_by_file_path_candidates(
        doc_status, Path("/app/data/inputs/README.md")
    )
    assert local["file_path"] == "README.md"


@pytest.mark.asyncio
async def test_pipeline_enqueue_file_stores_basename_not_full_path(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("LIGHTRAG_PARSER", raising=False)
    file_path = tmp_path / "report.pdf"
    file_path.write_bytes(b"pdf-bytes")
    rag = _EnqueueRag()

    success, track_id = await pipeline_enqueue_file(rag, file_path, "track-upload")

    assert success is True
    assert track_id == "track-upload"
    assert rag.enqueued[0]["file_paths"] == "report.pdf"


def test_delete_url_document_does_not_remove_local_readme(tmp_path):
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    local = input_dir / "README.md"
    local.write_text("uploaded readme", encoding="utf-8")

    for source in (_URL_A, f"{_URL_A}/"):
        stored_source = canonicalize_document_source(source)
        deleted, errors = delete_file_variants_by_file_path(
            input_dir, file_path=stored_source
        )

        assert deleted == []
        assert errors == []
        assert local.exists()

    deleted_local, errors_local = delete_file_variants_by_file_path(
        input_dir, file_path="README.md"
    )
    assert errors_local == []
    assert "README.md" in {Path(p).name for p in deleted_local}
    assert not local.exists()
