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

from taskcompendium.pipeline.review_transport import DEFAULT_MAX_BATCH_BYTES, BatchClient, batch_output

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
    keys = {}
    completed = {}
    misses = {}
    try:
        for request in requests:
            task_id = request["custom_id"]
            identity = {"format": 1, "model_revision": model_revision, "request": request}
            key = hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            keys[task_id] = key
        started = time.monotonic()
        # PersistentKvCache already treats unreadable storage as misses.
        saved_queries = cache.load_many(list(keys.values()))
        metrics.update_counter("review/cache/lookup_seconds", time.monotonic() - started)
        diagnostics = cache.read_diagnostics().blob_reads
        for name, value in (
            ("descriptor_seconds", diagnostics.descriptor_seconds),
            ("payload_seconds", diagnostics.payload_seconds),
            ("descriptor_lookups", diagnostics.descriptor_lookups),
            ("selected_shards", diagnostics.selected_shards),
            ("bytes_returned", diagnostics.bytes_returned),
        ):
            metrics.update_counter(f"review/cache/{name}", value)
        for request in requests:
            task_id = request["custom_id"]
            identity = {"format": 1, "model_revision": model_revision, "request": request}
            key = keys[task_id]
            saved = saved_queries.get(key)
            if saved is None:
                misses[task_id] = request
                continue
            try:
                envelope = json.loads(saved)
                if (
                    not isinstance(envelope, dict)
                    or envelope["identity"] != identity
                    or not isinstance(envelope["raw_output"], str)
                    or not valid_completion(envelope["raw_output"], task_id)
                ):
                    raise ValueError("Cached response does not match its request")
            except Exception:
                logger.warning("Ignoring invalid cached inference response for key %s", key)
                metrics.update_counter("review/cache/invalid_entries", 1)
                misses[task_id] = request
                continue
            completed[task_id] = envelope["raw_output"]
            (evidence / f"{key}.json").write_bytes(saved)
        metrics.update_counter("review/cache/hits", len(completed))
        metrics.update_counter("review/cache/misses", len(misses))
        if misses:
            missing_requests = list(misses.values())
            batch_identity = {"model_revision": model_revision, "requests": missing_requests}
            batch_key = hashlib.sha256(json.dumps(batch_identity, sort_keys=True).encode()).hexdigest()
            raw = submit(
                missing_requests,
                output_path / "submitted" / batch_key,
            )
            rows = {}
            for line in raw.splitlines():
                row = json.loads(line)
                task_id = row["custom_id"]
                if task_id not in misses:
                    raise ValueError(f"Unexpected batch response ID: {task_id}")
                rows.setdefault(task_id, []).append(line)
            for task_id, request in misses.items():
                result = "\n".join(rows.get(task_id, []))
                completed[task_id] = result
                if valid_completion(result, task_id):
                    metrics.update_counter("review/cache/valid_completions", 1)
                    identity = {"format": 1, "model_revision": model_revision, "request": request}
                    saved = json.dumps({"identity": identity, "raw_output": result}).encode()
                    started = time.monotonic()
                    try:
                        cache.store(keys[task_id], saved)
                    except Exception as error:
                        logger.warning("Inference cache store failed: %s", error)
                        metrics.update_counter("review/cache/store_failures", 1)
                    finally:
                        metrics.update_counter("review/cache/store_seconds", time.monotonic() - started)
                    (evidence / f"{keys[task_id]}.json").write_bytes(saved)
        return "\n".join(completed[task_id] for task_id in dict.fromkeys(row["custom_id"] for row in requests))
    finally:
        started = time.monotonic()
        try:
            cache.close()
        except Exception as error:
            logger.warning("Inference cache flush failed: %s", error)
            metrics.update_counter("review/cache/flush_failures", 1)
        finally:
            metrics.update_counter("review/cache/flush_seconds", time.monotonic() - started)
