# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exact request reuse backed by FineStore's persistent byte cache."""

import hashlib
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from finestore.cache import PersistentKvCache

from taskcompendium.pipeline.review_transport import BatchClient, batch_output


def cached_batch_output(
    client: BatchClient,
    requests: Sequence[dict[str, Any]],
    output_path: Path,
    *,
    cache_root: str,
    model_revision: str,
    poll_seconds: float,
    valid_completion: Callable[[str, str], bool],
) -> str:
    """Submit uncached queries and retain successful raw completions as evidence."""
    cache = PersistentKvCache.at(cache_root)
    evidence = output_path / "query-cache"
    evidence.mkdir(parents=True, exist_ok=True)
    keys = {}
    keyed_requests = []
    completed = {}
    misses = {}
    try:
        for request in requests:
            task_id = request["custom_id"]
            identity = {"format": 1, "model_revision": model_revision, "request": request}
            key = hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            keys[task_id] = key
            keyed_requests.append((request, identity, key))
        saved_by_key = cache.load_many([key for _request, _identity, key in keyed_requests])
        for request, identity, key in keyed_requests:
            task_id = request["custom_id"]
            saved = saved_by_key.get(key)
            if saved is None:
                misses[task_id] = request
                continue
            envelope = json.loads(saved)
            if envelope["identity"] != identity or not valid_completion(envelope["raw_output"], task_id):
                raise ValueError("Cached query completion does not match its validated request")
            completed[task_id] = envelope["raw_output"]
            (evidence / f"{key}.json").write_bytes(saved)
        if misses:
            missing_requests = list(misses.values())
            batch_identity = {"model_revision": model_revision, "requests": missing_requests}
            batch_key = hashlib.sha256(json.dumps(batch_identity, sort_keys=True).encode()).hexdigest()
            raw = batch_output(
                client,
                missing_requests,
                output_path / "submitted" / batch_key,
                filename="task-curation.jsonl",
                poll_seconds=poll_seconds,
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
                    identity = {"format": 1, "model_revision": model_revision, "request": request}
                    saved = json.dumps({"identity": identity, "raw_output": result}).encode()
                    cache.store(keys[task_id], saved)
                    (evidence / f"{keys[task_id]}.json").write_bytes(saved)
        return "\n".join(completed[task_id] for task_id in dict.fromkeys(row["custom_id"] for row in requests))
    finally:
        cache.close()
