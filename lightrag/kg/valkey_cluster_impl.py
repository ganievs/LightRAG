"""Valkey Cluster (Redis Cluster protocol) KV and doc-status storage.

``RedisKVStorage`` talks to a single Redis/Valkey endpoint through the plain
``redis.asyncio.Redis`` client, which does not follow ``MOVED``/``ASK``
redirects. This module provides the same two storages on top of
``redis.asyncio.cluster.RedisCluster``, so LightRAG can run against a sharded
Valkey Cluster (or Redis Cluster / ElastiCache with cluster mode enabled):

* the client discovers the slot map from ``VALKEY_CLUSTER_URI`` and routes
  every key to the owning shard, following redirects during resharding and
  failover;
* pipelines are cluster pipelines: commands are grouped per node and executed
  without ``MULTI``/``EXEC`` (this storage never relied on transactions);
* keyspace iteration uses ``scan_iter`` which walks every primary with its own
  cursor, because a plain ``SCAN`` cursor is meaningless across shards.

Key layout is identical to ``RedisKVStorage`` (``{workspace}_{namespace}:{id}``)
so data can be copied between the two backends key by key.
"""

from __future__ import annotations

import configparser
import json
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from collections.abc import AsyncIterator
from typing import Any, ClassVar, final

import pipmaster as pm

if not pm.is_installed("redis"):
    pm.install("redis")

from redis.asyncio.cluster import RedisCluster  # type: ignore
from redis.exceptions import ConnectionError, RedisError, TimeoutError  # type: ignore
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from lightrag.base import (
    BaseKVStorage,
    DocProcessingStatus,
    DocStatus,
    DocStatusStorage,
)
from lightrag.utils import (
    _cooperative_yield,
    get_pinyin_sort_key,
    logger,
    validate_workspace,
)

from ..kg.shared_storage import get_data_init_lock

config = configparser.ConfigParser()
config.read("config.ini", "utf-8")

DEFAULT_CLUSTER_URI = "redis://localhost:6379"

# Per-node connection pool size: RedisCluster keeps one pool per cluster node.
MAX_CONNECTIONS = int(os.getenv("VALKEY_CLUSTER_MAX_CONNECTIONS", "100"))
SOCKET_TIMEOUT = float(os.getenv("VALKEY_CLUSTER_SOCKET_TIMEOUT", "30.0"))
SOCKET_CONNECT_TIMEOUT = float(os.getenv("VALKEY_CLUSTER_CONNECT_TIMEOUT", "10.0"))
RETRY_ATTEMPTS = int(os.getenv("VALKEY_CLUSTER_RETRY_ATTEMPTS", "3"))
# Refuse to work while some hash slots have no reachable owner (safer default:
# a partially available cluster would silently lose part of the keyspace).
REQUIRE_FULL_COVERAGE = (
    os.getenv("VALKEY_CLUSTER_REQUIRE_FULL_COVERAGE", "true").strip().lower() == "true"
)
SCAN_BATCH_SIZE = int(os.getenv("VALKEY_CLUSTER_SCAN_BATCH_SIZE", "1000"))

valkey_retry = retry(
    stop=stop_after_attempt(RETRY_ATTEMPTS),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=(
        retry_if_exception_type(ConnectionError)
        | retry_if_exception_type(TimeoutError)
        | retry_if_exception_type(RedisError)
    ),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)


def _resolve_cluster_uri() -> str:
    return os.environ.get(
        "VALKEY_CLUSTER_URI",
        config.get("valkey", "cluster_uri", fallback=DEFAULT_CLUSTER_URI),
    )


class ValkeyClusterClientManager:
    """Share one ``RedisCluster`` client per URI between storage instances.

    A cluster client owns a connection pool per node plus the slot cache, so
    every storage instance (full_docs, text_chunks, doc_status, ...) reusing
    the same client avoids multiplying both.
    """

    _clients: ClassVar[dict[str, RedisCluster]] = {}
    _refs: ClassVar[dict[str, int]] = {}
    _lock = threading.Lock()

    @classmethod
    def get_client(cls, cluster_uri: str) -> RedisCluster:
        with cls._lock:
            if cluster_uri not in cls._clients:
                cls._clients[cluster_uri] = RedisCluster.from_url(
                    cluster_uri,
                    max_connections=MAX_CONNECTIONS,
                    decode_responses=True,
                    socket_timeout=SOCKET_TIMEOUT,
                    socket_connect_timeout=SOCKET_CONNECT_TIMEOUT,
                    require_full_coverage=REQUIRE_FULL_COVERAGE,
                )
                cls._refs[cluster_uri] = 0
                logger.info(f"Created shared Valkey Cluster client for {cluster_uri}")
            cls._refs[cluster_uri] += 1
            logger.debug(
                f"Valkey Cluster client {cluster_uri} reference count: {cls._refs[cluster_uri]}"
            )
            return cls._clients[cluster_uri]

    @classmethod
    async def release_client(cls, cluster_uri: str) -> None:
        client: RedisCluster | None = None
        with cls._lock:
            if cluster_uri in cls._refs:
                cls._refs[cluster_uri] -= 1
                logger.debug(
                    f"Valkey Cluster client {cluster_uri} reference count: {cls._refs[cluster_uri]}"
                )
                if cls._refs[cluster_uri] <= 0:
                    client = cls._clients.pop(cluster_uri, None)
                    del cls._refs[cluster_uri]
        if client is not None:
            try:
                await client.aclose()
                logger.info(
                    f"Closed Valkey Cluster client for {cluster_uri} (no more references)"
                )
            except Exception as e:
                logger.error(
                    f"Error closing Valkey Cluster client for {cluster_uri}: {e}"
                )

    @classmethod
    async def close_all_clients(cls) -> None:
        with cls._lock:
            clients = list(cls._clients.items())
            cls._clients.clear()
            cls._refs.clear()
        for uri, client in clients:
            try:
                await client.aclose()
                logger.info(f"Closed Valkey Cluster client for {uri}")
            except Exception as e:
                logger.error(f"Error closing Valkey Cluster client for {uri}: {e}")


class _ValkeyClusterStorageBase:
    """Connection handling and keyspace helpers shared by both storages."""

    namespace: str
    workspace: str
    final_namespace: str
    _cluster_uri: str
    _redis: RedisCluster | None
    _initialized: bool

    _empty_workspace_marker = ""
    _storage_kind = "storage"

    def _setup(self) -> None:
        validate_workspace(self.workspace)
        # VALKEY_WORKSPACE forces one workspace for every Valkey storage instance,
        # mirroring REDIS_WORKSPACE for the single-node backend.
        forced_workspace = os.environ.get("VALKEY_WORKSPACE")
        if forced_workspace and forced_workspace.strip():
            effective_workspace = forced_workspace.strip()
            logger.info(
                f"Using VALKEY_WORKSPACE environment variable: '{effective_workspace}' "
                f"(overriding '{self.workspace}/{self.namespace}')"
            )
        else:
            effective_workspace = self.workspace

        if effective_workspace:
            self.final_namespace = f"{effective_workspace}_{self.namespace}"
        else:
            self.final_namespace = self.namespace
            self.workspace = self._empty_workspace_marker

        self._cluster_uri = _resolve_cluster_uri()
        self._redis = None
        self._initialized = False

        try:
            self._redis = ValkeyClusterClientManager.get_client(self._cluster_uri)
            logger.info(
                f"[{self.workspace}] Initialized Valkey Cluster {self._storage_kind} "
                f"for {self.namespace} using shared cluster client"
            )
        except Exception as e:
            logger.error(
                f"[{self.workspace}] Failed to initialize Valkey Cluster {self._storage_kind}: {e}"
            )
            raise

    async def _connect(self) -> None:
        """Discover the cluster topology and verify every node answers."""
        try:
            async with self._get_connection() as redis:
                await redis.initialize()
                await redis.ping()
                try:
                    node_count = len(redis.get_nodes())
                except Exception:
                    node_count = "unknown"
                logger.info(
                    f"[{self.workspace}] Connected to Valkey Cluster ({node_count} nodes) "
                    f"for {self._storage_kind} namespace {self.namespace}"
                )
                self._initialized = True
        except Exception as e:
            logger.error(
                f"[{self.workspace}] Failed to connect to Valkey Cluster for "
                f"{self._storage_kind} {self.namespace}: {e}"
            )
            await self.close()
            raise

    @asynccontextmanager
    async def _get_connection(self):
        if not self._redis:
            raise ConnectionError("Valkey Cluster client not initialized")
        try:
            yield self._redis
        except ConnectionError as e:
            logger.error(
                f"[{self.workspace}] Valkey Cluster connection error in {self.namespace}: {e}"
            )
            raise
        except RedisError as e:
            logger.error(
                f"[{self.workspace}] Valkey Cluster operation error in {self.namespace}: {e}"
            )
            raise
        except Exception as e:
            logger.error(
                f"[{self.workspace}] Unexpected error in Valkey Cluster operation "
                f"for {self.namespace}: {e}"
            )
            raise

    async def close(self) -> None:
        """Release the shared client reference; the client closes with the last reference."""
        if getattr(self, "_redis", None) is not None:
            self._redis = None
        if getattr(self, "_cluster_uri", None):
            await ValkeyClusterClientManager.release_client(self._cluster_uri)
            logger.debug(
                f"[{self.workspace}] Released Valkey Cluster client reference for {self.namespace}"
            )

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    def _key(self, id: str) -> str:
        return f"{self.final_namespace}:{id}"

    def _id_from_key(self, key: str) -> str:
        return key.split(":", 1)[1]

    @property
    def _pattern(self) -> str:
        return f"{self.final_namespace}:*"

    async def _iter_key_batches(
        self, redis: RedisCluster, batch_size: int = SCAN_BATCH_SIZE
    ) -> AsyncIterator[list[str]]:
        """Yield namespace keys in batches.

        ``scan_iter`` on a cluster client walks every primary with a per-node
        cursor, which replaces the ``SCAN`` loops of the single-node backend.
        """
        batch: list[str] = []
        async for key in redis.scan_iter(match=self._pattern, count=batch_size):
            batch.append(key)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    async def _iter_items(self, redis: RedisCluster) -> AsyncIterator[tuple[str, str]]:
        """Yield ``(key, raw_json)`` for every record in the namespace."""
        async for keys in self._iter_key_batches(redis):
            pipe = redis.pipeline()
            for key in keys:
                pipe.get(key)
            values = await pipe.execute()
            for key, value in zip(keys, values):
                if value:
                    yield key, value

    async def _is_empty(self) -> bool:
        try:
            async with self._get_connection() as redis:
                async for _ in redis.scan_iter(match=self._pattern, count=1):
                    return False
                return True
        except Exception as e:
            logger.error(f"[{self.workspace}] Error checking if storage is empty: {e}")
            return True

    async def _drop_namespace(self) -> int:
        deleted_count = 0
        async with self._get_connection() as redis:
            async for keys in self._iter_key_batches(redis):
                pipe = redis.pipeline()
                for key in keys:
                    pipe.delete(key)
                results = await pipe.execute()
                deleted_count += sum(results)
        return deleted_count


@final
@dataclass
class ValkeyClusterKVStorage(_ValkeyClusterStorageBase, BaseKVStorage):
    _empty_workspace_marker = ""
    _storage_kind = "KV storage"

    def __post_init__(self):
        self._setup()

    async def initialize(self):
        """Connect to the cluster and migrate the legacy cache layout if needed."""
        async with get_data_init_lock():
            if self._initialized:
                return
            await self._connect()
            if self.namespace.endswith("_cache"):
                try:
                    await self._migrate_legacy_cache_structure()
                except Exception as e:
                    logger.error(
                        f"[{self.workspace}] Failed to migrate legacy cache structure: {e}"
                    )

    @valkey_retry
    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        async with self._get_connection() as redis:
            try:
                data = await redis.get(self._key(id))
                if data:
                    result = json.loads(data)
                    result.setdefault("create_time", 0)
                    result.setdefault("update_time", 0)
                    return result
                return None
            except json.JSONDecodeError as e:
                logger.error(f"[{self.workspace}] JSON decode error for id {id}: {e}")
                raise

    @valkey_retry
    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        if not ids:
            return []
        async with self._get_connection() as redis:
            try:
                pipe = redis.pipeline()
                for id in ids:
                    pipe.get(self._key(id))
                results = await pipe.execute()

                processed_results = []
                for result in results:
                    if result:
                        data = json.loads(result)
                        data.setdefault("create_time", 0)
                        data.setdefault("update_time", 0)
                        processed_results.append(data)
                    else:
                        processed_results.append(None)
                return processed_results
            except json.JSONDecodeError as e:
                logger.error(f"[{self.workspace}] JSON decode error in batch get: {e}")
                raise

    async def filter_keys(self, keys: set[str]) -> set[str]:
        if not keys:
            return set()
        async with self._get_connection() as redis:
            pipe = redis.pipeline()
            keys_list = list(keys)
            for key in keys_list:
                pipe.exists(self._key(key))
            results = await pipe.execute()
            existing_ids = {keys_list[i] for i, exists in enumerate(results) if exists}
            return set(keys) - existing_ids

    @valkey_retry
    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        if not data:
            return

        current_time = int(time.time())

        async with self._get_connection() as redis:
            try:
                pipe = redis.pipeline()
                for i, k in enumerate(data.keys(), start=1):
                    pipe.exists(self._key(k))
                    await _cooperative_yield(i)
                exists_results = await pipe.execute()

                for i, (k, v) in enumerate(data.items(), start=1):
                    if (
                        self.namespace.endswith("text_chunks")
                        and "llm_cache_list" not in v
                    ):
                        v["llm_cache_list"] = []

                    if exists_results[i - 1]:
                        v["update_time"] = current_time
                    else:
                        v["create_time"] = current_time
                        v["update_time"] = current_time

                    v["_id"] = k
                    await _cooperative_yield(i)

                pipe = redis.pipeline()
                for i, (k, v) in enumerate(data.items(), start=1):
                    pipe.set(self._key(k), json.dumps(v))
                    await _cooperative_yield(i)
                await pipe.execute()
            except json.JSONDecodeError as e:
                logger.error(f"[{self.workspace}] JSON decode error during upsert: {e}")
                raise

    async def index_done_callback(self) -> None:
        # Valkey persists on its own (AOF / RDB on every shard).
        pass

    async def is_empty(self) -> bool:
        return await self._is_empty()

    async def delete(self, ids: list[str]) -> None:
        if not ids:
            return
        async with self._get_connection() as redis:
            pipe = redis.pipeline()
            for id in ids:
                pipe.delete(self._key(id))
            results = await pipe.execute()
            deleted_count = sum(results)
            logger.info(
                f"[{self.workspace}] Deleted {deleted_count} of {len(ids)} entries from {self.namespace}"
            )

    async def drop(self) -> dict[str, str]:
        try:
            deleted_count = await self._drop_namespace()
            logger.info(
                f"[{self.workspace}] Dropped {deleted_count} keys from {self.namespace}"
            )
            return {"status": "success", "message": f"{deleted_count} keys dropped"}
        except Exception as e:
            logger.error(
                f"[{self.workspace}] Error dropping keys from {self.namespace}: {e}"
            )
            return {"status": "error", "message": str(e)}

    async def _migrate_legacy_cache_structure(self):
        """Flatten legacy nested LLM cache records (same rules as RedisKVStorage).

        Early exit as soon as a flattened ``mode:cache_type:hash`` key is seen.
        """
        from lightrag.utils import generate_cache_key

        async with self._get_connection() as redis:
            keys: list[str] = []
            async for batch in self._iter_key_batches(redis):
                keys.extend(batch)
            if not keys:
                return

            keys_to_migrate = []
            for key in keys:
                key_id = self._id_from_key(key)
                if ":" in key_id and len(key_id.split(":")) == 3:
                    logger.debug(
                        f"[{self.workspace}] Found flattened cache keys in {self.namespace}, skipping migration"
                    )
                    return

                data = await redis.get(key)
                if not data:
                    continue
                try:
                    parsed_data = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed_data, dict) and all(
                    isinstance(v, dict) and "return" in v for v in parsed_data.values()
                ):
                    keys_to_migrate.append((key, key_id, parsed_data))

            if not keys_to_migrate:
                return

            pipe = redis.pipeline()
            migration_count = 0
            for old_key, mode, nested_data in keys_to_migrate:
                pipe.delete(old_key)
                for cache_hash, cache_entry in nested_data.items():
                    cache_type = cache_entry.get("cache_type", "extract")
                    flattened_key = generate_cache_key(mode, cache_type, cache_hash)
                    pipe.set(self._key(flattened_key), json.dumps(cache_entry))
                    migration_count += 1
            await pipe.execute()

            if migration_count > 0:
                logger.info(
                    f"[{self.workspace}] Migrated {migration_count} legacy cache entries "
                    f"to flattened structure in Valkey Cluster"
                )


def _to_doc_status(doc_data: dict[str, Any]) -> DocProcessingStatus:
    data = doc_data.copy()
    data.pop("content", None)
    if "file_path" not in data:
        data["file_path"] = "no-file-path"
    if "metadata" not in data:
        data["metadata"] = {}
    if "error_msg" not in data:
        data["error_msg"] = None
    return DocProcessingStatus(**data)


@final
@dataclass
class ValkeyClusterDocStatusStorage(_ValkeyClusterStorageBase, DocStatusStorage):
    """Valkey Cluster implementation of document status storage."""

    _empty_workspace_marker = "_"
    _storage_kind = "doc status storage"

    def __post_init__(self):
        self._setup()

    async def initialize(self):
        async with get_data_init_lock():
            if self._initialized:
                return
            await self._connect()

    async def filter_keys(self, keys: set[str]) -> set[str]:
        if not keys:
            return set()
        async with self._get_connection() as redis:
            pipe = redis.pipeline()
            keys_list = list(keys)
            for key in keys_list:
                pipe.exists(self._key(key))
            results = await pipe.execute()
            existing_ids = {keys_list[i] for i, exists in enumerate(results) if exists}
            return set(keys) - existing_ids

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        if not ids:
            return []
        ordered_results: list[dict[str, Any] | None] = []
        async with self._get_connection() as redis:
            try:
                pipe = redis.pipeline()
                for id in ids:
                    pipe.get(self._key(id))
                results = await pipe.execute()
                for result_data in results:
                    if result_data:
                        try:
                            ordered_results.append(json.loads(result_data))
                        except json.JSONDecodeError as e:
                            logger.error(
                                f"[{self.workspace}] JSON decode error in get_by_ids: {e}"
                            )
                            raise
                    else:
                        ordered_results.append(None)
            except Exception as e:
                logger.error(f"[{self.workspace}] Error in get_by_ids: {e}")
                raise
        return ordered_results

    async def get_status_counts(self) -> dict[str, int]:
        counts = {status.value: 0 for status in DocStatus}
        async with self._get_connection() as redis:
            try:
                async for _, value in self._iter_items(redis):
                    try:
                        status = json.loads(value).get("status")
                    except json.JSONDecodeError:
                        continue
                    if status in counts:
                        counts[status] += 1
            except Exception as e:
                logger.error(f"[{self.workspace}] Error getting status counts: {e}")
        return counts

    async def get_docs_by_status(
        self, status: DocStatus
    ) -> dict[str, DocProcessingStatus]:
        return await self.get_docs_by_statuses([status])

    async def get_docs_by_statuses(
        self, statuses: list[DocStatus]
    ) -> dict[str, DocProcessingStatus]:
        """Fetch every document whose status is in ``statuses`` in one keyspace pass."""
        if not statuses:
            return {}
        status_values = {s.value for s in statuses}
        result: dict[str, DocProcessingStatus] = {}
        async with self._get_connection() as redis:
            try:
                async for key, value in self._iter_items(redis):
                    try:
                        doc_data = json.loads(value)
                        if doc_data.get("status") not in status_values:
                            continue
                        result[self._id_from_key(key)] = _to_doc_status(doc_data)
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.error(
                            f"[{self.workspace}] Error processing document {key}: {e}"
                        )
                        continue
            except Exception as e:
                logger.error(
                    f"[{self.workspace}] SCAN interrupted while fetching docs by statuses "
                    f"— result is incomplete ({len(result)} documents collected): {e!r}"
                )
                raise
        return result

    async def get_docs_by_track_id(
        self, track_id: str
    ) -> dict[str, DocProcessingStatus]:
        result: dict[str, DocProcessingStatus] = {}
        async with self._get_connection() as redis:
            try:
                async for key, value in self._iter_items(redis):
                    try:
                        doc_data = json.loads(value)
                        if doc_data.get("track_id") != track_id:
                            continue
                        result[self._id_from_key(key)] = _to_doc_status(doc_data)
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.error(
                            f"[{self.workspace}] Error processing document {key}: {e}"
                        )
                        continue
            except Exception as e:
                logger.error(f"[{self.workspace}] Error getting docs by track_id: {e}")
        return result

    async def index_done_callback(self) -> None:
        pass

    async def is_empty(self) -> bool:
        return await self._is_empty()

    @valkey_retry
    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        if not data:
            return
        logger.debug(
            f"[{self.workspace}] Inserting {len(data)} records to {self.namespace}"
        )
        async with self._get_connection() as redis:
            try:
                for i, (doc_id, doc_data) in enumerate(data.items(), start=1):
                    if "chunks_list" not in doc_data:
                        doc_data["chunks_list"] = []
                    await _cooperative_yield(i)

                pipe = redis.pipeline()
                for i, (k, v) in enumerate(data.items(), start=1):
                    pipe.set(self._key(k), json.dumps(v))
                    await _cooperative_yield(i)
                await pipe.execute()
            except json.JSONDecodeError as e:
                logger.error(f"[{self.workspace}] JSON decode error during upsert: {e}")
                raise

    @valkey_retry
    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        async with self._get_connection() as redis:
            try:
                data = await redis.get(self._key(id))
                return json.loads(data) if data else None
            except json.JSONDecodeError as e:
                logger.error(f"[{self.workspace}] JSON decode error for id {id}: {e}")
                raise

    async def delete(self, doc_ids: list[str]) -> None:
        if not doc_ids:
            return
        async with self._get_connection() as redis:
            pipe = redis.pipeline()
            for doc_id in doc_ids:
                pipe.delete(self._key(doc_id))
            results = await pipe.execute()
            deleted_count = sum(results)
            logger.info(
                f"[{self.workspace}] Deleted {deleted_count} of {len(doc_ids)} "
                f"doc status entries from {self.namespace}"
            )

    async def get_docs_paginated(
        self,
        status_filter: DocStatus | None = None,
        status_filters: list[DocStatus] | None = None,
        page: int = 1,
        page_size: int = 50,
        sort_field: str = "updated_at",
        sort_direction: str = "desc",
    ) -> tuple[list[tuple[str, DocProcessingStatus]], int]:
        """Load, filter and sort in memory; the keyspace is walked shard by shard."""
        status_filter_values = self.resolve_status_filter_values(
            status_filter=status_filter,
            status_filters=status_filters,
        )

        page = max(page, 1)
        if page_size < 10:
            page_size = 10
        elif page_size > 200:
            page_size = 200
        if sort_field not in ["created_at", "updated_at", "id", "file_path"]:
            sort_field = "updated_at"
        if sort_direction.lower() not in ["asc", "desc"]:
            sort_direction = "desc"

        all_docs: list[tuple[str, DocProcessingStatus, Any]] = []
        async with self._get_connection() as redis:
            try:
                async for key, value in self._iter_items(redis):
                    try:
                        doc_data = json.loads(value)
                        if (
                            status_filter_values is not None
                            and doc_data.get("status") not in status_filter_values
                        ):
                            continue
                        doc_id = self._id_from_key(key)
                        doc_status = _to_doc_status(doc_data)
                        if sort_field == "id":
                            sort_key = doc_id
                        elif sort_field == "file_path":
                            sort_key = get_pinyin_sort_key(
                                getattr(doc_status, "file_path", "") or ""
                            )
                        else:
                            sort_key = doc_data.get(sort_field, "")
                        all_docs.append((doc_id, doc_status, sort_key))
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.error(
                            f"[{self.workspace}] Error processing document {key}: {e}"
                        )
                        continue
            except Exception as e:
                logger.error(f"[{self.workspace}] Error getting paginated docs: {e}")
                return [], 0

        reverse_sort = sort_direction.lower() == "desc"
        all_docs.sort(key=lambda x: x[2], reverse=reverse_sort)
        docs = [(doc_id, doc_status) for doc_id, doc_status, _ in all_docs]
        total_count = len(docs)
        start_idx = (page - 1) * page_size
        return docs[start_idx : start_idx + page_size], total_count

    async def get_all_status_counts(self) -> dict[str, int]:
        counts = await self.get_status_counts()
        counts["all"] = sum(counts.values())
        return counts

    async def _find_first(
        self, predicate, what: str
    ) -> tuple[str, dict[str, Any]] | None:
        async with self._get_connection() as redis:
            try:
                async for key, value in self._iter_items(redis):
                    try:
                        doc_data = json.loads(value)
                    except json.JSONDecodeError as e:
                        logger.error(
                            f"[{self.workspace}] JSON decode error in {what}: {e}"
                        )
                        continue
                    if predicate(doc_data):
                        return self._id_from_key(key), doc_data
                return None
            except Exception as e:
                logger.error(f"[{self.workspace}] Error in {what}: {e}")
                return None

    async def get_doc_by_file_path(self, file_path: str) -> dict[str, Any] | None:
        found = await self._find_first(
            lambda d: d.get("file_path") == file_path, "get_doc_by_file_path"
        )
        return found[1] if found else None

    async def get_doc_by_file_basename(
        self, basename: str
    ) -> tuple[str, dict[str, Any]] | None:
        """Exact match on the canonical basename stored in ``file_path``."""
        if not basename or basename == "unknown_source":
            return None
        return await self._find_first(
            lambda d: d.get("file_path") == basename, "get_doc_by_file_basename"
        )

    async def get_doc_by_content_hash(
        self, content_hash: str
    ) -> tuple[str, dict[str, Any]] | None:
        if not content_hash:
            return None
        return await self._find_first(
            lambda d: d.get("content_hash") == content_hash, "get_doc_by_content_hash"
        )

    async def drop(self) -> dict[str, str]:
        try:
            deleted_count = await self._drop_namespace()
            logger.info(
                f"[{self.workspace}] Dropped {deleted_count} doc status keys from {self.namespace}"
            )
            return {"status": "success", "message": "data dropped"}
        except Exception as e:
            logger.error(
                f"[{self.workspace}] Error dropping doc status {self.namespace}: {e}"
            )
            return {"status": "error", "message": str(e)}
