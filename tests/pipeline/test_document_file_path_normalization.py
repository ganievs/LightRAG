import sys

import pytest

sys.argv = sys.argv[:1]

from lightrag.api.routers.document_routes import (  # noqa: E402
    DocStatusResponse,
    is_local_basename_file_path,
    normalize_file_path,
    pipeline_index_texts,
    stored_file_path_for_response,
)
from lightrag.utils_pipeline import canonicalize_document_source  # noqa: E402
from lightrag.base import DocStatus  # noqa: E402
from lightrag.constants import PROCESS_OPTION_CHUNK_FIXED  # noqa: E402
from lightrag.pipeline import _PipelineMixin  # noqa: E402


class DummyRAG:
    def __init__(self):
        self.enqueued_calls = []
        self.processed = False
        # _resolve_text_chunking reads addon_params; {} -> default chunker config.
        self.addon_params = {}

    async def apipeline_enqueue_documents(
        self,
        input,
        file_paths=None,
        track_id=None,
        process_options=None,
        chunk_options=None,
    ):
        self.enqueued_calls.append(
            {
                "input": input,
                "file_paths": file_paths,
                "track_id": track_id,
                "process_options": process_options,
                "chunk_options": chunk_options,
            }
        )

    async def apipeline_process_enqueue_documents(self):
        self.processed = True


class CaptureDocStatus:
    def __init__(self):
        self.upserts = []

    async def upsert(self, data):
        self.upserts.append(data)


class DummyPipeline(_PipelineMixin):
    def __init__(self):
        self.doc_status = CaptureDocStatus()


class CaptureKV:
    def __init__(self):
        self.upserts = []

    async def filter_keys(self, keys):
        return set(keys)

    async def upsert(self, data):
        self.upserts.append(data)


@pytest.mark.asyncio
async def test_pipeline_index_texts_rejects_missing_file_sources():
    rag = DummyRAG()

    with pytest.raises(ValueError, match="valid file source"):
        await pipeline_index_texts(
            rag,
            texts=["alpha"],
            file_sources=[None],
            track_id="track-1",
        )

    assert rag.enqueued_calls == []
    assert rag.processed is False


@pytest.mark.asyncio
async def test_pipeline_index_texts_preserves_file_source_path():
    rag = DummyRAG()

    await pipeline_index_texts(
        rag,
        texts=["alpha"],
        file_sources=["/tmp/source/alpha.txt"],
        track_id="track-1",
    )

    assert len(rag.enqueued_calls) == 1
    call = rag.enqueued_calls[0]
    assert call["input"] == ["alpha"]
    assert call["file_paths"] == ["/tmp/source/alpha.txt"]
    assert call["track_id"] == "track-1"
    assert call["process_options"] == PROCESS_OPTION_CHUNK_FIXED
    # No chunking config supplied -> default F snapshot from addon_params.
    assert isinstance(call["chunk_options"], dict)
    assert "fixed_token" in call["chunk_options"]
    assert rag.processed is True


def test_doc_status_response_uses_non_null_unknown_source():
    response = DocStatusResponse(
        id="doc-1",
        content_summary="summary",
        content_length=5,
        status=DocStatus.PENDING,
        created_at="2026-03-19T00:00:00+00:00",
        updated_at="2026-03-19T00:00:00+00:00",
        file_path=normalize_file_path(None),
    )

    assert response.file_path == "unknown_source"


@pytest.mark.asyncio
async def test_error_document_enqueue_canonicalizes_file_path_before_upsert():
    rag = DummyPipeline()

    await rag.apipeline_enqueue_error_documents(
        [
            {
                "file_path": "/tmp/uploads/report.[native-Fi].pdf",
                "error_description": "bad file",
                "original_error": "parse failed",
            }
        ],
        track_id="track-1",
    )

    saved = next(iter(rag.doc_status.upserts[0].values()))
    assert saved["file_path"] == "/tmp/uploads/report.pdf"


@pytest.mark.asyncio
async def test_custom_chunks_use_canonical_unknown_source_before_upsert():
    from lightrag import LightRAG

    rag = LightRAG.__new__(LightRAG)
    rag.full_docs = CaptureKV()
    rag.text_chunks = CaptureKV()
    rag.chunks_vdb = CaptureKV()
    rag.tokenizer = type("Tokenizer", (), {"encode": lambda self, text: [text]})()

    async def _process_extract_entities(chunks):
        return []

    async def _insert_done():
        return None

    rag._process_extract_entities = _process_extract_entities
    rag._insert_done = _insert_done

    await rag.ainsert_custom_chunks("full text", ["chunk text"], doc_id="doc-1")

    assert rag.full_docs.upserts[0]["doc-1"]["file_path"] == "unknown_source"
    chunk = next(iter(rag.text_chunks.upserts[0].values()))
    assert chunk["file_path"] == "unknown_source"


@pytest.mark.parametrize(
    "source, expected",
    [
        (
            "https://gitlab.example.com/group/repo-a/-/blob/main/README.md",
            "https://gitlab.example.com/group/repo-a/-/blob/main/README.md",
        ),
        (
            "s3://bucket/project-a/docs/index.md",
            "s3://bucket/project-a/docs/index.md",
        ),
        ("project-a/docs/index.md", "project-a/docs/index.md"),
        ("notes.[native].md", "notes.md"),
        ("https://x/notes.[native].md", "https://x/notes.md"),
        ("HTTPS://X/notes.[native].md", "HTTPS://X/notes.md"),
        (
            "https://x/notes.[native].md?ref=main#section",
            "https://x/notes.md?ref=main#section",
        ),
        (
            "https://x/notes.[native].md?v=1.2",
            "https://x/notes.[native].md?v=1.2",
        ),
        ("https://x/notes.[native].md/", "https://x/notes.[native].md/"),
        ("https://x/README.md/", "https://x/README.md/"),
        ("https://x/", "https://x/"),
        ("https://x/unknown_source", "https://x/unknown_source"),
        (r"C:\docs\notes.[native].md", r"C:\docs\notes.md"),
        ("abc.docx", "abc.docx"),
        ("/tmp/sub/abc.docx", "/tmp/sub/abc.docx"),
        ("/tmp/sub/", "/tmp/sub/"),
        (None, "unknown_source"),
        ("", "unknown_source"),
        ("no-file-path", "unknown_source"),
        ("  ", "unknown_source"),
    ],
)
def test_canonicalize_document_source_identity(source, expected):
    assert canonicalize_document_source(source) == expected


def test_stored_file_path_for_response_keeps_url_and_maps_sentinels():
    url = "https://gitlab.example.com/group/repo/-/blob/main/README.md"
    assert stored_file_path_for_response(url) == url
    assert stored_file_path_for_response(None) == "unknown_source"
    assert stored_file_path_for_response("") == "unknown_source"
    assert stored_file_path_for_response("no-file-path") == "unknown_source"
    assert stored_file_path_for_response("README.md") == "README.md"


def test_is_local_basename_file_path():
    assert is_local_basename_file_path("README.md")
    assert is_local_basename_file_path("report.pdf")
    assert is_local_basename_file_path("notes.[native].md")
    assert not is_local_basename_file_path(
        "https://gitlab.example.com/group/repo/-/blob/main/README.md"
    )
    assert not is_local_basename_file_path("s3://bucket/docs/index.md")
    assert not is_local_basename_file_path("project-a/docs/index.md")
    assert not is_local_basename_file_path("/tmp/sub/abc.docx")
    assert not is_local_basename_file_path("README.md/")
    assert not is_local_basename_file_path("")


@pytest.mark.asyncio
async def test_pipeline_index_texts_rejects_duplicate_sources():
    rag = DummyRAG()

    with pytest.raises(ValueError, match="File sources must be unique"):
        await pipeline_index_texts(
            rag,
            texts=["alpha", "beta"],
            file_sources=[
                "https://gitlab.example.com/group/repo/-/blob/main/README.md",
                "https://gitlab.example.com/group/repo/-/blob/main/README.md",
            ],
        )

    assert rag.enqueued_calls == []


@pytest.mark.asyncio
async def test_pipeline_index_texts_keeps_distinct_urls_with_same_basename():
    rag = DummyRAG()
    sources = [
        "https://gitlab.example.com/group/repo-a/-/blob/main/README.md",
        "https://gitlab.example.com/group/repo-b/-/blob/main/README.md",
    ]

    await pipeline_index_texts(
        rag,
        texts=["alpha from repo A", "beta from repo B"],
        file_sources=sources,
        track_id="track-urls",
    )

    assert rag.enqueued_calls[0]["file_paths"] == sources
    assert rag.processed is True


@pytest.mark.asyncio
async def test_pipeline_index_texts_strips_hint_from_url_last_segment():
    rag = DummyRAG()

    await pipeline_index_texts(
        rag,
        texts=["alpha"],
        file_sources=["https://x/notes.[native].md"],
        track_id="track-hint",
    )

    assert rag.enqueued_calls[0]["file_paths"] == ["https://x/notes.md"]


@pytest.mark.asyncio
async def test_pipeline_index_texts_rejects_hinted_and_plain_url_as_duplicate():
    rag = DummyRAG()

    with pytest.raises(ValueError, match="File sources must be unique"):
        await pipeline_index_texts(
            rag,
            texts=["alpha", "beta"],
            file_sources=[
                "https://x/notes.[native].md",
                "https://x/notes.md",
            ],
        )

    assert rag.enqueued_calls == []
