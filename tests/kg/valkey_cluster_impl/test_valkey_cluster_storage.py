"""Offline tests for the Valkey Cluster KV and doc-status storages.

The cluster client is replaced with an in-memory fake that mirrors the part of
the ``redis.asyncio.cluster.RedisCluster`` surface the storages use:
``initialize``, ``ping``, ``get_nodes``, ``scan_iter``, ``get`` and a
non-transactional pipeline with ``get``/``set``/``exists``/``delete``.

The fake also asserts the cluster contract: a plain ``scan`` cursor is never
used (it is per node on a cluster) and pipelines never ask for a transaction.
"""

from __future__ import annotations

import json

import pytest

from lightrag.base import DocStatus
from lightrag.kg import (
    STORAGE_ENV_REQUIREMENTS,
    STORAGE_IMPLEMENTATIONS,
    STORAGES,
    verify_storage_implementation,
)
from lightrag.namespace import NameSpace

pytestmark = pytest.mark.offline


class _DummyEmbeddingFunc:
    embedding_dim = 1
    max_token_size = 1

    async def __call__(self, texts, **kwargs):
        return [[0.0] for _ in texts]


def _doc(
    status: str,
    file_path: str,
    content_hash: str | None = None,
    track_id: str | None = None,
    updated_at: str = "2024-01-01T00:00:00+00:00",
) -> dict:
    payload = {
        "content_summary": f"{status} summary",
        "content_length": 10,
        "file_path": file_path,
        "status": status,
        "created_at": "2024-01-01T00:00:00+00:00",
        "updated_at": updated_at,
        "metadata": {},
        "error_msg": None,
    }
    if content_hash is not None:
        payload["content_hash"] = content_hash
    if track_id is not None:
        payload["track_id"] = track_id
    return payload


class _FakeClusterPipeline:
    """Non-transactional pipeline: commands are queued and run in order."""

    def __init__(self, store: dict[str, str]):
        self._store = store
        self._ops: list[tuple] = []

    def get(self, key: str):
        self._ops.append(("get", key))
        return self

    def set(self, key: str, value: str):
        self._ops.append(("set", key, value))
        return self

    def exists(self, key: str):
        self._ops.append(("exists", key))
        return self

    def delete(self, key: str):
        self._ops.append(("delete", key))
        return self

    async def execute(
        self, raise_on_error: bool = True, allow_redirections: bool = True
    ):
        results = []
        for op in self._ops:
            kind = op[0]
            if kind == "get":
                results.append(self._store.get(op[1]))
            elif kind == "set":
                self._store[op[1]] = op[2]
                results.append(True)
            elif kind == "exists":
                results.append(1 if op[1] in self._store else 0)
            elif kind == "delete":
                existed = op[1] in self._store
                self._store.pop(op[1], None)
                results.append(1 if existed else 0)
        self._ops.clear()
        return results


class _FakeRedisCluster:
    def __init__(self, nodes: int = 6):
        self.store: dict[str, str] = {}
        self._nodes = nodes
        self.initialized = False
        self.closed = False
        self.scan_iter_calls = 0

    async def initialize(self):
        self.initialized = True
        return self

    async def ping(self):
        return True

    def get_nodes(self):
        return [object() for _ in range(self._nodes)]

    async def scan(self, *args, **kwargs):  # pragma: no cover - contract guard
        raise AssertionError(
            "plain SCAN must not be used on a cluster client; use scan_iter"
        )

    def scan_iter(self, match=None, count=None, **kwargs):
        self.scan_iter_calls += 1
        prefix = match[:-1] if match and match.endswith("*") else match
        keys = [k for k in list(self.store) if prefix is None or k.startswith(prefix)]

        async def _aiter():
            for k in keys:
                yield k

        return _aiter()

    def pipeline(self, transaction=None, shard_hint=None):
        assert not transaction, "cluster pipelines do not support transactions"
        return _FakeClusterPipeline(self.store)

    async def get(self, key: str):
        return self.store.get(key)

    async def aclose(self):
        self.closed = True


@pytest.fixture
def shared_data():
    from lightrag.kg.shared_storage import (
        finalize_share_data,
        initialize_share_data,
    )

    initialize_share_data()
    try:
        yield
    finally:
        finalize_share_data()


@pytest.fixture
def fake_cluster(monkeypatch):
    fake = _FakeRedisCluster()
    released: list[str] = []

    monkeypatch.setattr(
        "lightrag.kg.valkey_cluster_impl.ValkeyClusterClientManager.get_client",
        classmethod(lambda cls, uri: fake),
    )

    async def _release(cls, uri):
        released.append(uri)

    monkeypatch.setattr(
        "lightrag.kg.valkey_cluster_impl.ValkeyClusterClientManager.release_client",
        classmethod(_release),
    )
    fake.released = released
    return fake


@pytest.fixture
def kv_storage(fake_cluster):
    from lightrag.kg.valkey_cluster_impl import ValkeyClusterKVStorage

    storage = ValkeyClusterKVStorage(
        namespace=NameSpace.KV_STORE_TEXT_CHUNKS,
        global_config={},
        embedding_func=_DummyEmbeddingFunc(),
        workspace="test",
    )
    return storage


@pytest.fixture
def doc_status(fake_cluster):
    from lightrag.kg.valkey_cluster_impl import ValkeyClusterDocStatusStorage

    storage = ValkeyClusterDocStatusStorage(
        namespace=NameSpace.DOC_STATUS,
        global_config={},
        embedding_func=_DummyEmbeddingFunc(),
        workspace="test",
    )
    storage._initialized = True
    return storage


def _store_raw(storage, doc_id: str, payload: dict) -> None:
    storage._redis.store[f"{storage.final_namespace}:{doc_id}"] = json.dumps(payload)


# --- registry -----------------------------------------------------------------


def test_storages_are_registered():
    assert (
        "ValkeyClusterKVStorage"
        in STORAGE_IMPLEMENTATIONS["KV_STORAGE"]["implementations"]
    )
    assert (
        "ValkeyClusterDocStatusStorage"
        in STORAGE_IMPLEMENTATIONS["DOC_STATUS_STORAGE"]["implementations"]
    )
    assert STORAGE_ENV_REQUIREMENTS["ValkeyClusterKVStorage"] == ["VALKEY_CLUSTER_URI"]
    assert STORAGE_ENV_REQUIREMENTS["ValkeyClusterDocStatusStorage"] == [
        "VALKEY_CLUSTER_URI"
    ]
    assert STORAGES["ValkeyClusterKVStorage"] == ".kg.valkey_cluster_impl"
    assert STORAGES["ValkeyClusterDocStatusStorage"] == ".kg.valkey_cluster_impl"
    verify_storage_implementation("KV_STORAGE", "ValkeyClusterKVStorage")
    verify_storage_implementation("DOC_STATUS_STORAGE", "ValkeyClusterDocStatusStorage")


def test_registered_classes_import_and_satisfy_required_methods():
    import importlib

    module = importlib.import_module(
        STORAGES["ValkeyClusterKVStorage"], package="lightrag"
    )
    for storage_type, name in (
        ("KV_STORAGE", "ValkeyClusterKVStorage"),
        ("DOC_STATUS_STORAGE", "ValkeyClusterDocStatusStorage"),
    ):
        cls = getattr(module, name)
        for method in STORAGE_IMPLEMENTATIONS[storage_type]["required_methods"]:
            assert callable(getattr(cls, method))


# --- key layout and workspace --------------------------------------------------


def test_key_layout_matches_redis_backend(kv_storage, doc_status):
    assert kv_storage.final_namespace == "test_text_chunks"
    assert kv_storage._key("chunk-1") == "test_text_chunks:chunk-1"
    assert doc_status.final_namespace == "test_doc_status"


def test_valkey_workspace_env_overrides_workspace(monkeypatch, fake_cluster):
    monkeypatch.setenv("VALKEY_WORKSPACE", "forced")
    from lightrag.kg.valkey_cluster_impl import ValkeyClusterKVStorage

    storage = ValkeyClusterKVStorage(
        namespace=NameSpace.KV_STORE_FULL_DOCS,
        global_config={},
        embedding_func=_DummyEmbeddingFunc(),
        workspace="ignored",
    )
    assert storage.final_namespace == "forced_full_docs"


async def test_initialize_connects_and_close_releases_shared_client(
    shared_data, kv_storage
):
    await kv_storage.initialize()
    assert kv_storage._redis.initialized is True
    assert kv_storage._initialized is True
    fake = kv_storage._redis
    await kv_storage.close()
    assert kv_storage._redis is None
    assert fake.released == [kv_storage._cluster_uri]


# --- KV storage ---------------------------------------------------------------


async def test_kv_upsert_sets_timestamps_and_chunk_defaults(kv_storage):
    await kv_storage.upsert({"c1": {"content": "hello"}})
    stored = json.loads(kv_storage._redis.store["test_text_chunks:c1"])
    assert stored["_id"] == "c1"
    assert stored["llm_cache_list"] == []
    assert stored["create_time"] == stored["update_time"] > 0

    first_create = stored["create_time"]
    await kv_storage.upsert({"c1": {"content": "hello again"}})
    updated = json.loads(kv_storage._redis.store["test_text_chunks:c1"])
    assert updated["content"] == "hello again"
    assert "create_time" not in updated or updated.get("create_time") in (
        first_create,
        0,
    )
    assert updated["update_time"] >= first_create


async def test_kv_get_by_id_and_get_by_ids_preserve_order(kv_storage):
    await kv_storage.upsert({"a": {"v": 1}, "b": {"v": 2}})
    assert (await kv_storage.get_by_id("a"))["v"] == 1
    assert await kv_storage.get_by_id("missing") is None
    results = await kv_storage.get_by_ids(["b", "missing", "a"])
    assert [r["v"] if r else None for r in results] == [2, None, 1]
    assert await kv_storage.get_by_ids([]) == []


async def test_kv_filter_keys_returns_only_unknown_ids(kv_storage):
    await kv_storage.upsert({"a": {"v": 1}})
    assert await kv_storage.filter_keys({"a", "b", "c"}) == {"b", "c"}
    assert await kv_storage.filter_keys(set()) == set()


async def test_kv_delete_is_empty_and_drop(kv_storage):
    assert await kv_storage.is_empty() is True
    await kv_storage.upsert({"a": {"v": 1}, "b": {"v": 2}, "c": {"v": 3}})
    assert await kv_storage.is_empty() is False

    await kv_storage.delete(["a", "missing"])
    assert "test_text_chunks:a" not in kv_storage._redis.store
    assert "test_text_chunks:b" in kv_storage._redis.store

    # a key from another namespace must survive the drop
    kv_storage._redis.store["test_full_docs:x"] = json.dumps({"v": 0})
    result = await kv_storage.drop()
    assert result == {"status": "success", "message": "2 keys dropped"}
    assert list(kv_storage._redis.store) == ["test_full_docs:x"]
    assert kv_storage._redis.scan_iter_calls >= 3


async def test_kv_scan_batches_cover_more_than_one_batch(kv_storage, monkeypatch):
    monkeypatch.setattr("lightrag.kg.valkey_cluster_impl.SCAN_BATCH_SIZE", 3)
    await kv_storage.upsert({f"k{i}": {"v": i} for i in range(10)})
    assert (await kv_storage.drop())["message"] == "10 keys dropped"


async def test_kv_legacy_cache_migration_flattens_nested_entries(
    shared_data, fake_cluster
):
    from lightrag.kg.valkey_cluster_impl import ValkeyClusterKVStorage

    storage = ValkeyClusterKVStorage(
        namespace=NameSpace.KV_STORE_LLM_RESPONSE_CACHE,
        global_config={},
        embedding_func=_DummyEmbeddingFunc(),
        workspace="test",
    )
    nested = {
        "hash1": {"return": "answer-1", "cache_type": "query"},
        "hash2": {"return": "answer-2"},
    }
    storage._redis.store["test_llm_response_cache:hybrid"] = json.dumps(nested)

    await storage.initialize()

    store = storage._redis.store
    assert "test_llm_response_cache:hybrid" not in store
    assert (
        json.loads(store["test_llm_response_cache:hybrid:query:hash1"])["return"]
        == "answer-1"
    )
    assert (
        json.loads(store["test_llm_response_cache:hybrid:extract:hash2"])["return"]
        == "answer-2"
    )


# --- doc status storage -------------------------------------------------------


async def test_doc_status_upsert_get_and_counts(doc_status):
    await doc_status.upsert(
        {
            "d1": _doc(DocStatus.PROCESSED.value, "a.pdf"),
            "d2": _doc(DocStatus.FAILED.value, "b.pdf"),
            "d3": _doc(DocStatus.PROCESSED.value, "c.pdf"),
        }
    )
    stored = await doc_status.get_by_id("d1")
    assert stored["chunks_list"] == []
    assert await doc_status.get_by_id("nope") is None

    counts = await doc_status.get_status_counts()
    assert counts[DocStatus.PROCESSED.value] == 2
    assert counts[DocStatus.FAILED.value] == 1
    assert counts[DocStatus.PENDING.value] == 0
    assert (await doc_status.get_all_status_counts())["all"] == 3

    by_ids = await doc_status.get_by_ids(["d2", "missing"])
    assert by_ids[0]["file_path"] == "b.pdf" and by_ids[1] is None
    assert await doc_status.filter_keys({"d1", "new"}) == {"new"}


async def test_doc_status_get_docs_by_statuses_and_track_id(doc_status):
    _store_raw(
        doc_status, "d1", _doc(DocStatus.PROCESSED.value, "a.pdf", track_id="t1")
    )
    _store_raw(doc_status, "d2", _doc(DocStatus.FAILED.value, "b.pdf", track_id="t1"))
    _store_raw(doc_status, "d3", _doc(DocStatus.PENDING.value, "c.pdf", track_id="t2"))
    # a record without content_summary cannot become a DocProcessingStatus
    doc_status._redis.store[f"{doc_status.final_namespace}:broken"] = "{not json"

    processed = await doc_status.get_docs_by_status(DocStatus.PROCESSED)
    assert set(processed) == {"d1"}
    assert processed["d1"].status == DocStatus.PROCESSED

    both = await doc_status.get_docs_by_statuses(
        [DocStatus.PROCESSED, DocStatus.FAILED]
    )
    assert set(both) == {"d1", "d2"}
    assert await doc_status.get_docs_by_statuses([]) == {}

    tracked = await doc_status.get_docs_by_track_id("t1")
    assert set(tracked) == {"d1", "d2"}


async def test_doc_status_paginated_sorting_filtering_and_bounds(doc_status):
    for i in range(12):
        status = DocStatus.PROCESSED.value if i % 2 == 0 else DocStatus.FAILED.value
        _store_raw(
            doc_status,
            f"d{i:02d}",
            _doc(
                status,
                f"f{i:02d}.pdf",
                updated_at=f"2024-01-{i + 1:02d}T00:00:00+00:00",
            ),
        )

    docs, total = await doc_status.get_docs_paginated(page=1, page_size=10)
    assert total == 12
    assert len(docs) == 10
    assert docs[0][0] == "d11"  # newest first for updated_at desc

    docs, total = await doc_status.get_docs_paginated(
        status_filter=DocStatus.FAILED, page=1, page_size=10, sort_direction="asc"
    )
    assert total == 6
    assert [doc_id for doc_id, _ in docs] == ["d01", "d03", "d05", "d07", "d09", "d11"]

    docs, total = await doc_status.get_docs_paginated(
        page=2, page_size=5, sort_field="id", sort_direction="asc"
    )
    assert total == 12
    # page_size floors at 10, so page 2 holds the last two ids
    assert [doc_id for doc_id, _ in docs] == ["d10", "d11"]

    docs, total = await doc_status.get_docs_paginated(
        page=1, page_size=10, sort_field="id", sort_direction="asc"
    )
    assert [doc_id for doc_id, _ in docs][:5] == ["d00", "d01", "d02", "d03", "d04"]

    docs, total = await doc_status.get_docs_paginated(
        page=0, page_size=1, sort_field="bogus", sort_direction="sideways"
    )
    assert total == 12 and len(docs) == 10  # page_size floors at 10, page at 1


async def test_doc_status_lookups_by_path_basename_and_hash(doc_status):
    _store_raw(doc_status, "d1", _doc(DocStatus.PROCESSED.value, "report.pdf", "h1"))
    _store_raw(doc_status, "legacy", _doc(DocStatus.PROCESSED.value, "old.pdf"))
    _store_raw(
        doc_status, "unsourced", _doc(DocStatus.PROCESSED.value, "unknown_source")
    )

    assert (await doc_status.get_doc_by_file_path("report.pdf"))["content_hash"] == "h1"
    assert await doc_status.get_doc_by_file_path("nope.pdf") is None

    assert (await doc_status.get_doc_by_file_basename("report.pdf"))[0] == "d1"
    assert await doc_status.get_doc_by_file_basename("") is None
    assert await doc_status.get_doc_by_file_basename("unknown_source") is None

    assert (await doc_status.get_doc_by_content_hash("h1"))[0] == "d1"
    assert await doc_status.get_doc_by_content_hash("") is None
    assert await doc_status.get_doc_by_content_hash("zzz") is None


async def test_doc_status_delete_and_drop(doc_status):
    _store_raw(doc_status, "d1", _doc(DocStatus.PROCESSED.value, "a.pdf"))
    _store_raw(doc_status, "d2", _doc(DocStatus.PROCESSED.value, "b.pdf"))
    doc_status._redis.store["test_text_chunks:c1"] = json.dumps({"v": 1})

    await doc_status.delete(["d1"])
    assert await doc_status.get_by_id("d1") is None
    assert await doc_status.is_empty() is False

    assert await doc_status.drop() == {"status": "success", "message": "data dropped"}
    assert await doc_status.is_empty() is True
    assert "test_text_chunks:c1" in doc_status._redis.store
