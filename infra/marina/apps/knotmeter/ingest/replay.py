# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# Marin additions use Apache-2.0; the adaptation is distributed under GPL-3.0-only.
# Adapted from InferenceX 315128ce compute-request-timeline.ts; see THIRD_PARTY.md.

"""Retain replay ancestry and integer timestamps from AIPerf records."""

import json
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any, TypedDict


class RequestRecord(TypedDict):
    id: str
    conversation_id: str
    corpus_id: str
    source_outer_idx: int
    source_inner_idx: int | None
    source_kind: str
    turn: int
    phase: str
    replay_index: int | None
    root_correlation_id: str | None
    parent_correlation_id: str | None
    correlation_id: str | None
    worker_id: str
    agent_depth: int
    credit_ns: int
    start_ns: int
    ack_ns: int | None
    end_ns: int
    ttft_ms: float | None
    tpot_ms: float | None
    input_tokens: int | None
    output_tokens: int | None
    cancelled: bool
    error: str | None


def profile_records(path: Path) -> Iterator[dict[str, Any]]:
    with path.open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def prepare_requests(records: Callable[[], Iterable[dict[str, Any]]]) -> tuple[int, list[RequestRecord]]:
    """Rank stable replay roots and subtract the origin before JSON number conversion."""
    parents: dict[str, str] = {}
    roots: dict[str, str] = {}
    root_first: dict[str, int] = {}
    session_first: dict[str, int] = {}
    origin: int | None = None
    for record in records():
        meta = record["metadata"]
        session = meta.get("x_correlation_id")
        dispatch = meta.get("credit_issued_ns")
        if dispatch is None:
            dispatch = meta["request_start_ns"]
        origin = dispatch if origin is None else min(origin, dispatch)
        root = meta.get("root_correlation_id")
        if root:
            root_first[root] = min(root_first.get(root, dispatch), dispatch)
        elif session:
            session_first[session] = min(session_first.get(session, dispatch), dispatch)
        if session and meta.get("root_correlation_id"):
            roots[session] = meta["root_correlation_id"]
        if session and meta.get("parent_correlation_id"):
            parents[session] = meta["parent_correlation_id"]

    def root_of(meta: dict[str, Any]) -> str | None:
        if meta.get("root_correlation_id"):
            return meta["root_correlation_id"]
        session = meta.get("x_correlation_id")
        current = session
        seen = set()
        while current and current not in seen:
            seen.add(current)
            if current in roots:
                return roots[current]
            if current not in parents:
                return current
            current = parents[current]
        return session

    if origin is None:
        raise ValueError("Profile contains no requests")
    first_dispatch = dict(root_first)
    for session, dispatch in session_first.items():
        root = root_of({"x_correlation_id": session})
        if root is not None:
            first_dispatch[root] = min(first_dispatch.get(root, dispatch), dispatch)
    replay_index = {root: i for i, root in enumerate(sorted(first_dispatch, key=lambda r: (first_dispatch[r], r)))}
    requests: list[RequestRecord] = []
    for record in records():
        meta, metrics = record["metadata"], record["metrics"]
        dispatch = meta.get("credit_issued_ns")
        if dispatch is None:
            dispatch = meta["request_start_ns"]
        requests.append(
            {
                "id": meta["x_request_id"],
                "conversation_id": meta["conversation_id"],
                "corpus_id": meta["source_trace_id"],
                "source_outer_idx": meta["source_outer_idx"],
                "source_inner_idx": meta.get("source_inner_idx"),
                "source_kind": meta["source_kind"],
                "turn": meta["turn_index"],
                "phase": meta["benchmark_phase"],
                "replay_index": replay_index.get(root_of(meta)),
                "root_correlation_id": meta.get("root_correlation_id"),
                "parent_correlation_id": meta.get("parent_correlation_id"),
                "correlation_id": meta.get("x_correlation_id"),
                "worker_id": meta.get("worker_id", "unknown"),
                "agent_depth": meta.get("agent_depth", 0),
                "credit_ns": dispatch - origin,
                "start_ns": meta["request_start_ns"] - origin,
                "ack_ns": meta["request_ack_ns"] - origin if meta.get("request_ack_ns") is not None else None,
                "end_ns": meta["request_end_ns"] - origin,
                "ttft_ms": metrics.get("time_to_first_token", {}).get("value"),
                "tpot_ms": metrics.get("time_per_output_token", metrics.get("inter_token_latency", {})).get("value"),
                "input_tokens": metrics.get("input_sequence_length", {}).get("value"),
                "output_tokens": metrics.get("output_sequence_length", {}).get("value"),
                "cancelled": meta.get("was_cancelled", False),
                "error": record.get("error", {}).get("type"),
            }
        )
    requests.sort(key=lambda r: r["start_ns"])
    return origin, requests
