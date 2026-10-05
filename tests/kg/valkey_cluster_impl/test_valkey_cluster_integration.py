"""Integration tests against a real Valkey / Redis Cluster.

Skipped unless pytest runs with ``--run-integration`` and ``VALKEY_CLUSTER_URI``
points at a cluster. ``valkey-cluster.compose.yaml`` next to this file starts a
local 6-node cluster; see its header for the exact commands (the test must run
inside the compose network). The keys below are chosen without hash tags, so a 3-shard
cluster spreads them over several nodes and the test exercises ``MOVED``
routing, cross-node pipelines and per-node ``scan_iter``.
"""

from __future__ import annotations

import os
import uuid

import pytest

from lightrag.base import DocStatus
from lightrag.namespace import NameSpace

pytestmark = pytest.mark.integration

CLUSTER_URI = os.environ.get("VALKEY_CLUSTER_URI")


class _DummyEmbeddingFunc:
    embedding_dim = 1
    max_token_size = 1

    async def __call__(self, texts, **kwargs):
        return [[0.0] for _ in texts]


@pytest.fixture
async def storages():
    if not CLUSTER_URI:
        pytest.skip("VALKEY_CLUSTER_URI is not set")
    from lightrag.kg.shared_storage import (
        finalize_share_data,
        initialize_share_data,
    )
    from lightrag.kg.valkey_cluster_impl import (
        ValkeyClusterDocStatusStorage,
        ValkeyClusterKVStorage,
    )

    initialize_share_data()
    workspace = f"it_{uuid.uuid4().hex[:8]}"
    kv = ValkeyClusterKVStorage(
        namespace=NameSpace.KV_STORE_TEXT_CHUNKS,
        global_config={},
        embedding_func=_DummyEmbeddingFunc(),
        workspace=workspace,
    )
    ds = ValkeyClusterDocStatusStorage(
        namespace=NameSpace.DOC_STATUS,
        global_config={},
        embedding_func=_DummyEmbeddingFunc(),
        workspace=workspace,
    )
    await kv.initialize()
    await ds.initialize()
    try:
        yield kv, ds
    finally:
        await kv.drop()
        await ds.drop()
        await kv.close()
        await ds.close()
        finalize_share_data()


async def test_cluster_round_trip_across_slots(storages):
    kv, ds = storages
    assert len(kv._redis.get_nodes()) >= 3

    chunks = {f"chunk-{i}": {"content": f"text {i}"} for i in range(200)}
    await kv.upsert(chunks)
    assert await kv.is_empty() is False

    # the keys must really be spread over several shards, otherwise this test
    # would not prove anything about MOVED routing
    owners = {
        kv._redis.get_node_from_key(kv._key(f"chunk-{i}")).name for i in range(200)
    }
    assert len(owners) >= 2, owners

    rows = await kv.get_by_ids([f"chunk-{i}" for i in range(200)])
    assert [r["content"] for r in rows] == [f"text {i}" for i in range(200)]
    assert await kv.filter_keys({"chunk-1", "chunk-999"}) == {"chunk-999"}

    await ds.upsert(
        {
            f"doc-{i}": {
                "content_summary": "s",
                "content_length": 1,
                "file_path": f"f{i}.pdf",
                "status": DocStatus.PROCESSED.value
                if i % 2
                else DocStatus.PENDING.value,
                "created_at": "2024-01-01T00:00:00+00:00",
                "updated_at": f"2024-01-{i + 1:02d}T00:00:00+00:00",
            }
            for i in range(20)
        }
    )
    counts = await ds.get_status_counts()
    assert counts[DocStatus.PROCESSED.value] == 10
    assert counts[DocStatus.PENDING.value] == 10
    docs, total = await ds.get_docs_paginated(page=1, page_size=10)
    assert total == 20 and docs[0][0] == "doc-19"
    assert (await ds.get_doc_by_file_path("f3.pdf"))["file_path"] == "f3.pdf"

    await kv.delete([f"chunk-{i}" for i in range(100)])
    assert (await kv.drop())["message"] == "100 keys dropped"
