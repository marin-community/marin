# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exact request reuse backed by FineStore's persistent byte cache."""

import hashlib
import json
import logging
import time
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path
from typing import Any

from finestore.cache import PersistentKvCache
from zephyr import counters

from taskcompendium.pipeline.review_requests import DEFAULT_MAX_BATCH_BYTES, BatchClient, batch_output

logger = logging.getLogger(__name__)


def cached_batch_output(
    client: BatchClient,
    requests: Sequence[dict[str, Any]],
    output_path: Path,
    *,
    cache_root: str,
    model_revision: str,
    poll_seconds: float,
    valid_completion: Callable[[str, str], bool],
    max_batch_bytes: int = DEFAULT_MAX_BATCH_BYTES,
) -> str:
    return cached_request_output(
        requests,
        output_path,
        cache_root=cache_root,
        model_revision=model_revision,
        valid_completion=valid_completion,
        submit=partial(
            batch_output,
            client,
            filename="task-curation.jsonl",
            poll_seconds=poll_seconds,
            max_batch_bytes=max_batch_bytes,
        ),
    )


def _request_identity(request: dict[str, Any], model_revision: str) -> dict[str, Any]:
    return {"format": 1, "model_revision": model_revision, "request": request}


def _cache_key(identity: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _record_lookup_diagnostics(cache: PersistentKvCache, lookup_seconds: float) -> None:
    metrics = counters.current_stage()
    metrics.update_counter("review/cache/lookup_seconds", lookup_seconds)
    diagnostics = cache.read_diagnostics().blob_reads
    for name, value in (
        ("descriptor_seconds", diagnostics.descriptor_seconds),
        ("payload_seconds", diagnostics.payload_seconds),
        ("descriptor_lookups", diagnostics.descriptor_lookups),
        ("selected_shards", diagnostics.selected_shards),
        ("bytes_returned", diagnostics.bytes_returned),
    ):
        metrics.update_counter(f"review/cache/{name}", value)


def _cached_output(
    saved: bytes, identity: dict[str, Any], task_id: str, valid_completion: Callable[[str, str], bool]
) -> str:
    """The raw output of a cached envelope, which must hold this exact request and a valid completion."""
    envelope = json.loads(saved)
    if (
        not isinstance(envelope, dict)
        or envelope["identity"] != identity
        or not isinstance(envelope["raw_output"], str)
        or not valid_completion(envelope["raw_output"], task_id)
    ):
        raise ValueError("Cached response does not match its request")
    return envelope["raw_output"]


def _cached_completions(
    cache: PersistentKvCache,
    requests: Sequence[dict[str, Any]],
    keys: dict[str, str],
    *,
    model_revision: str,
    valid_completion: Callable[[str, str], bool],
    evidence: Path,
) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    """Valid cached outputs by task ID, copied to ``evidence``, and the requests the cache cannot answer."""
    metrics = counters.current_stage()
    started = time.monotonic()
    # PersistentKvCache already treats unreadable storage as misses.
    saved_queries = cache.load_many(list(keys.values()))
    _record_lookup_diagnostics(cache, time.monotonic() - started)
    completed = {}
    misses = {}
    for request in requests:
        task_id = request["custom_id"]
        key = keys[task_id]
        saved = saved_queries.get(key)
        if saved is None:
            misses[task_id] = request
            continue
        try:
            completed[task_id] = _cached_output(
                saved, _request_identity(request, model_revision), task_id, valid_completion
            )
        except Exception:
            logger.warning("Ignoring invalid cached inference response for key %s", key)
            metrics.update_counter("review/cache/invalid_entries", 1)
            misses[task_id] = request
            continue
        (evidence / f"{key}.json").write_bytes(saved)
    return completed, misses


def _submitted_completions(
    misses: dict[str, dict[str, Any]],
    output_path: Path,
    *,
    model_revision: str,
    submit: Callable[[Sequence[dict[str, Any]], Path], str],
) -> dict[str, str]:
    """Submit the missed requests together and return each one's raw output lines, empty when it has none."""
    missing_requests = list(misses.values())
    batch_identity = {"model_revision": model_revision, "requests": missing_requests}
    batch_key = hashlib.sha256(json.dumps(batch_identity, sort_keys=True).encode()).hexdigest()
    raw = submit(missing_requests, output_path / "submitted" / batch_key)
    rows = {}
    for line in raw.splitlines():
        task_id = json.loads(line)["custom_id"]
        if task_id not in misses:
            raise ValueError(f"Unexpected batch response ID: {task_id}")
        rows.setdefault(task_id, []).append(line)
    return {task_id: "\n".join(rows.get(task_id, [])) for task_id in misses}


def _store_completion(cache: PersistentKvCache, key: str, identity: dict[str, Any], output: str, evidence: Path) -> None:
    """Cache a valid completion on a best-effort basis and retain it as evidence."""
    metrics = counters.current_stage()
    metrics.update_counter("review/cache/valid_completions", 1)
    saved = json.dumps({"identity": identity, "raw_output": output}).encode()
    started = time.monotonic()
    try:
        cache.store(key, saved)
    except Exception as error:
        logger.warning("Inference cache store failed: %s", error)
        metrics.update_counter("review/cache/store_failures", 1)
    finally:
        metrics.update_counter("review/cache/store_seconds", time.monotonic() - started)
    (evidence / f"{key}.json").write_bytes(saved)


def _close_cache(cache: PersistentKvCache) -> None:
    metrics = counters.current_stage()
    started = time.monotonic()
    try:
        cache.close()
    except Exception as error:
        logger.warning("Inference cache flush failed: %s", error)
        metrics.update_counter("review/cache/flush_failures", 1)
    finally:
        metrics.update_counter("review/cache/flush_seconds", time.monotonic() - started)


def cached_request_output(
    requests: Sequence[dict[str, Any]],
    output_path: Path,
    *,
    cache_root: str,
    model_revision: str,
    valid_completion: Callable[[str, str], bool],
    submit: Callable[[Sequence[dict[str, Any]], Path], str],
) -> str:
    """Submit uncached queries and retain successful raw completions as evidence."""
    metrics = counters.current_stage()
    metrics.update_counter("review/cache/request_observations", len(requests))
    cache = PersistentKvCache.at(cache_root)
    evidence = output_path / "query-cache"
    evidence.mkdir(parents=True, exist_ok=True)
    try:
        keys = {request["custom_id"]: _cache_key(_request_identity(request, model_revision)) for request in requests}
        completed, misses = _cached_completions(
            cache,
            requests,
            keys,
            model_revision=model_revision,
            valid_completion=valid_completion,
            evidence=evidence,
        )
        metrics.update_counter("review/cache/hits", len(completed))
        metrics.update_counter("review/cache/misses", len(misses))
        if misses:
            submitted = _submitted_completions(misses, output_path, model_revision=model_revision, submit=submit)
            for task_id, output in submitted.items():
                completed[task_id] = output
                if valid_completion(output, task_id):
                    identity = _request_identity(misses[task_id], model_revision)
                    _store_completion(cache, keys[task_id], identity, output, evidence)
        return "\n".join(completed[task_id] for task_id in dict.fromkeys(row["custom_id"] for row in requests))
    finally:
        _close_cache(cache)
