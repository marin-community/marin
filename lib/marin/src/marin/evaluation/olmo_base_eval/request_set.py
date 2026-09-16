# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The frozen, model-independent request-set artifact.

A request set is the exact ``(context, continuation)`` gold scoring pairs that
OLMo-Eval would score for BPB, captured once (offline, from OLMo-Eval) so the
Marin runtime never depends on OLMo-Eval or SC. It is stored as a single
newline-delimited JSON file of instance records plus a ``manifest.json`` sidecar
that records provenance and per-task instance counts (the cheap parity check
against the SC oracle's ``num_instances``).

Layout under a request-set directory ``<dir>/``:
  - ``requests.jsonl`` : one ``{task, doc_id, context, continuation}`` per line
  - ``manifest.json``  : ``{version, olmo_eval_git_sha, source, tasks: {task: count}}``
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass

import fsspec

REQUEST_SCHEMA_VERSION = "1"
REQUESTS_FILENAME = "requests.jsonl"
MANIFEST_FILENAME = "manifest.json"


@dataclass(frozen=True)
class RequestInstance:
    """One gold scoring instance: a continuation conditioned on a context."""

    task: str
    doc_id: int
    context: str
    continuation: str


@dataclass(frozen=True)
class RequestSetManifest:
    """Provenance and per-task instance counts for a request set."""

    version: str
    olmo_eval_git_sha: str | None
    source: str
    tasks: dict[str, int]


def write_request_set(
    directory: str,
    instances: list[RequestInstance],
    *,
    olmo_eval_git_sha: str | None,
    source: str,
) -> RequestSetManifest:
    """Write ``requests.jsonl`` + ``manifest.json`` under ``directory``.

    ``directory`` may be a local path or a ``gs://`` URL.
    """
    counts: dict[str, int] = defaultdict(int)
    requests_path = f"{directory.rstrip('/')}/{REQUESTS_FILENAME}"
    with fsspec.open(requests_path, "w") as handle:
        for instance in instances:
            handle.write(json.dumps(asdict(instance)) + "\n")
            counts[instance.task] += 1

    manifest = RequestSetManifest(
        version=REQUEST_SCHEMA_VERSION,
        olmo_eval_git_sha=olmo_eval_git_sha,
        source=source,
        tasks=dict(counts),
    )
    manifest_path = f"{directory.rstrip('/')}/{MANIFEST_FILENAME}"
    with fsspec.open(manifest_path, "w") as handle:
        handle.write(json.dumps(asdict(manifest), indent=2))
    return manifest


def read_manifest(directory: str) -> RequestSetManifest:
    """Read the request-set manifest from ``directory``."""
    manifest_path = f"{directory.rstrip('/')}/{MANIFEST_FILENAME}"
    with fsspec.open(manifest_path, "r") as handle:
        payload = json.load(handle)
    if payload.get("version") != REQUEST_SCHEMA_VERSION:
        raise ValueError(f"unsupported request-set version {payload.get('version')!r} at {directory}")
    return RequestSetManifest(**payload)


def load_request_set(directory: str) -> dict[str, list[RequestInstance]]:
    """Load a request set as ``{task: [RequestInstance, ...]}``, ordered by doc_id appearance.

    Validates the loaded per-task counts against the manifest so a truncated or
    partially written artifact fails loudly rather than silently dropping
    instances (which would corrupt the per-task BPB mean).
    """
    manifest = read_manifest(directory)
    requests_path = f"{directory.rstrip('/')}/{REQUESTS_FILENAME}"
    by_task: dict[str, list[RequestInstance]] = defaultdict(list)
    with fsspec.open(requests_path, "r") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            by_task[record["task"]].append(RequestInstance(**record))

    loaded_counts = {task: len(instances) for task, instances in by_task.items()}
    if loaded_counts != manifest.tasks:
        raise ValueError(f"request-set count mismatch at {directory}: manifest={manifest.tasks} loaded={loaded_counts}")
    return dict(by_task)
