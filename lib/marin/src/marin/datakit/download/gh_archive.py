# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Download and normalize held-out GH Archive event slices for PPL/gap evals."""

from __future__ import annotations

import gzip
import io
import json
import logging
import posixpath
import re
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import requests
from rigging.filesystem import open_url
from zephyr.writers import atomic_rename

from marin.execution import ExecutorStep
from marin.execution.step_spec import StepSpec
from marin.utils import fsspec_exists, fsspec_mkdirs

logger = logging.getLogger(__name__)

GH_ARCHIVE_BASE_URL = "https://data.gharchive.org"
GH_ARCHIVE_REQUIRED_EVENT_TYPES: tuple[str, ...] = (
    "PushEvent",
    "PullRequestEvent",
    "IssuesEvent",
    "IssueCommentEvent",
)
GH_ARCHIVE_OPTIONAL_EVENT_TYPES: tuple[str, ...] = ("WorkflowRunEvent",)
GH_ARCHIVE_DEFAULT_EVENT_TYPES: tuple[str, ...] = (*GH_ARCHIVE_REQUIRED_EVENT_TYPES, *GH_ARCHIVE_OPTIONAL_EVENT_TYPES)

HASH_KEY_FIELDS = frozenset({"sha", "before", "after", "head", "head_sha", "base_sha", "tree_id", "commit_id"})
ID_KEY_FIELDS = frozenset(
    {
        "id",
        "node_id",
        "actor_id",
        "repo_id",
        "comment_id",
        "issue_id",
        "pull_request_id",
        "run_id",
    }
)
TIMESTAMP_KEY_FIELDS = frozenset(
    {"created_at", "updated_at", "closed_at", "merged_at", "run_started_at", "run_completed_at", "published_at"}
)

HEX_RE = re.compile(r"^[0-9a-f]{16,}$", flags=re.IGNORECASE)
SHA_RE = re.compile(r"^[0-9a-f]{7,64}$", flags=re.IGNORECASE)
LONG_INT_RE = re.compile(r"^\d{8,}$")
UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", flags=re.IGNORECASE)
TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]{24,}$")
ISO_8601_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$")


@dataclass(frozen=True)
class GhArchiveDownloadConfig:
    output_path: str
    start_date: str
    end_date: str
    start_hour: int = 0
    end_hour: int = 23
    event_types: tuple[str, ...] = GH_ARCHIVE_DEFAULT_EVENT_TYPES
    max_events_per_event_type: int | None = None
    request_timeout: int = 120
    base_url: str = GH_ARCHIVE_BASE_URL
    metadata_filename: str = "metadata.json"
    skip_existing: bool = True


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"Expected date in YYYY-MM-DD format, got {value!r}") from exc


def _validate_download_config(cfg: GhArchiveDownloadConfig) -> None:
    start = _parse_date(cfg.start_date)
    end = _parse_date(cfg.end_date)
    if start > end:
        raise ValueError(f"start_date must be <= end_date, got {cfg.start_date} > {cfg.end_date}")
    if not 0 <= cfg.start_hour <= 23:
        raise ValueError(f"start_hour must be in [0, 23], got {cfg.start_hour}")
    if not 0 <= cfg.end_hour <= 23:
        raise ValueError(f"end_hour must be in [0, 23], got {cfg.end_hour}")
    if cfg.start_hour > cfg.end_hour:
        raise ValueError(f"start_hour must be <= end_hour, got {cfg.start_hour} > {cfg.end_hour}")
    if not cfg.event_types:
        raise ValueError("event_types must include at least one event type")
    if len(set(cfg.event_types)) != len(cfg.event_types):
        raise ValueError("event_types must be unique")
    if cfg.max_events_per_event_type is not None and cfg.max_events_per_event_type <= 0:
        raise ValueError("max_events_per_event_type must be positive")
    if cfg.request_timeout <= 0:
        raise ValueError("request_timeout must be positive")


def gh_archive_hour_urls(
    *,
    start_date: str,
    end_date: str,
    start_hour: int = 0,
    end_hour: int = 23,
    base_url: str = GH_ARCHIVE_BASE_URL,
) -> list[str]:
    start = _parse_date(start_date)
    end = _parse_date(end_date)
    if start > end:
        raise ValueError(f"start_date must be <= end_date, got {start_date} > {end_date}")
    if start_hour > end_hour:
        raise ValueError(f"start_hour must be <= end_hour, got {start_hour} > {end_hour}")
    if not 0 <= start_hour <= 23 or not 0 <= end_hour <= 23:
        raise ValueError("start_hour and end_hour must be in [0, 23]")

    urls: list[str] = []
    current = start
    while current <= end:
        day = current.isoformat()
        for hour in range(start_hour, end_hour + 1):
            urls.append(f"{base_url.rstrip('/')}/{day}-{hour}.json.gz")
        current += timedelta(days=1)
    return urls


def _bucket_identifier(value: str, *, force: bool = False) -> str:
    if LONG_INT_RE.fullmatch(value):
        return f"<INT_{len(value)}>"
    if UUID_RE.fullmatch(value):
        return "<UUID>"
    if HEX_RE.fullmatch(value):
        return f"<HEX_{len(value)}>"
    if SHA_RE.fullmatch(value):
        return f"<SHA_{len(value)}>"
    if TOKEN_RE.fullmatch(value):
        return f"<ID_{len(value)}>"
    if force:
        return "<ID>"
    return value


def _bucket_timestamp(value: str) -> str:
    if ISO_8601_RE.fullmatch(value):
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return f"<DATE:{timestamp.date().isoformat()}>"
    return value


def _bucket_hash(value: str) -> str:
    if SHA_RE.fullmatch(value) or HEX_RE.fullmatch(value):
        return f"<SHA_{len(value)}>"
    return "<SHA>"


def _bucket_url(value: str) -> str:
    try:
        parsed = urlsplit(value)
    except ValueError:
        return "<URL>"
    if not parsed.scheme or not parsed.netloc:
        return value

    bucketed_segments = [_bucket_identifier(segment) for segment in parsed.path.split("/")]

    query_pairs = []
    for key, query_value in parse_qsl(parsed.query, keep_blank_values=True):
        bucketed_query_value = _bucket_scalar(query_value, path=(key,))
        if not isinstance(bucketed_query_value, str):
            bucketed_query_value = str(bucketed_query_value)
        query_pairs.append((key, bucketed_query_value))

    return urlunsplit(
        (
            parsed.scheme,
            parsed.netloc,
            "/".join(bucketed_segments),
            urlencode(query_pairs, doseq=True),
            parsed.fragment,
        )
    )


def _bucket_scalar(value: str, *, path: tuple[str, ...]) -> str:
    key = path[-1] if path else ""

    if key in HASH_KEY_FIELDS:
        return _bucket_hash(value)
    if key in ID_KEY_FIELDS:
        return _bucket_identifier(value, force=True)
    if key in TIMESTAMP_KEY_FIELDS:
        return _bucket_timestamp(value)
    if value.startswith(("https://", "http://")):
        return _bucket_url(value)
    if ISO_8601_RE.fullmatch(value):
        return _bucket_timestamp(value)
    if LONG_INT_RE.fullmatch(value):
        return f"<INT_{len(value)}>"
    if SHA_RE.fullmatch(value):
        return f"<SHA_{len(value)}>"
    return value


def _mask_json_value(value: Any, *, path: tuple[str, ...]) -> Any:
    if isinstance(value, dict):
        return {key: _mask_json_value(item, path=(*path, key)) for key, item in value.items()}
    if isinstance(value, list):
        return [_mask_json_value(item, path=path) for item in value]
    if isinstance(value, str):
        return _bucket_scalar(value, path=path)
    if isinstance(value, int) and path and path[-1] in ID_KEY_FIELDS:
        return f"<INT_{len(str(abs(value)))}>"
    return value


def stable_json_serialize(value: Any) -> str:
    """Serialize JSON with deterministic key order and separators."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def normalize_gh_archive_event(
    event: dict[str, Any],
    *,
    event_types: set[str],
) -> tuple[str, dict[str, str]] | None:
    event_type = event.get("type")
    if not isinstance(event_type, str):
        return None
    if event_type not in event_types:
        return None

    masked_event = _mask_json_value(event, path=())
    return event_type, {"text": stable_json_serialize(masked_event)}


def read_gh_archive_hour(url: str, timeout: int) -> Iterator[dict[str, Any]]:
    """Yield GH Archive events from one hourly ``.json.gz`` file."""
    with requests.get(url, timeout=timeout, stream=True) as response:
        if response.status_code == 404:
            logger.info("GH Archive hour not found, skipping: %s", url)
            return
        response.raise_for_status()
        with gzip.GzipFile(fileobj=response.raw) as gz_file:
            with io.TextIOWrapper(gz_file, encoding="utf-8") as reader:
                for line_number, line in enumerate(reader, start=1):
                    text = line.strip()
                    if not text:
                        continue
                    try:
                        payload = json.loads(text)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Invalid JSON in {url} line {line_number}") from exc
                    if not isinstance(payload, dict):
                        continue
                    yield payload


def _event_output_path(output_path: str, event_type: str) -> str:
    return posixpath.join(output_path, event_type, "part-00000.jsonl.gz")


def _all_event_type_caps_reached(counts: dict[str, int], cap: int | None) -> bool:
    if cap is None:
        return False
    return all(value >= cap for value in counts.values())


def _write_metadata(path: str, payload: dict[str, Any]) -> None:
    with atomic_rename(path) as temp_path:
        with open_url(temp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)


def download_gh_archive_events(
    cfg: GhArchiveDownloadConfig,
    *,
    read_hour_events: Callable[[str, int], Iterable[dict[str, Any]]] = read_gh_archive_hour,
) -> dict[str, Any]:
    _validate_download_config(cfg)

    event_types = tuple(cfg.event_types)
    counts = {event_type: 0 for event_type in event_types}
    output_files = {event_type: _event_output_path(cfg.output_path, event_type) for event_type in event_types}
    metadata_path = posixpath.join(cfg.output_path, cfg.metadata_filename)

    if cfg.skip_existing and all(fsspec_exists(path) for path in output_files.values()) and fsspec_exists(metadata_path):
        logger.info("Skipping GH Archive download; output already exists at %s", cfg.output_path)
        with open_url(metadata_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        return {
            "success": True,
            "skipped": True,
            "counts": metadata.get("counts", counts),
            "output_files": output_files,
        }

    scanned_hour_urls: list[str] = []
    event_type_set = set(event_types)

    with ExitStack() as stack:
        writers: dict[str, Any] = {}
        for event_type, output_file in output_files.items():
            fsspec_mkdirs(posixpath.dirname(output_file), exist_ok=True)
            temp_path = stack.enter_context(atomic_rename(output_file))
            writers[event_type] = stack.enter_context(open_url(temp_path, "wt", encoding="utf-8", compression="gzip"))

        for hour_url in gh_archive_hour_urls(
            start_date=cfg.start_date,
            end_date=cfg.end_date,
            start_hour=cfg.start_hour,
            end_hour=cfg.end_hour,
            base_url=cfg.base_url,
        ):
            scanned_hour_urls.append(hour_url)
            for event in read_hour_events(hour_url, cfg.request_timeout):
                normalized = normalize_gh_archive_event(event, event_types=event_type_set)
                if normalized is None:
                    continue
                event_type, row = normalized
                if cfg.max_events_per_event_type is not None and counts[event_type] >= cfg.max_events_per_event_type:
                    continue
                json.dump(row, writers[event_type], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                writers[event_type].write("\n")
                counts[event_type] += 1
            if _all_event_type_caps_reached(counts, cfg.max_events_per_event_type):
                break

    metadata = {
        "start_date": cfg.start_date,
        "end_date": cfg.end_date,
        "start_hour": cfg.start_hour,
        "end_hour": cfg.end_hour,
        "base_url": cfg.base_url,
        "event_types": event_types,
        "max_events_per_event_type": cfg.max_events_per_event_type,
        "counts": counts,
        "output_files": output_files,
        "hours_scanned": scanned_hour_urls,
    }
    _write_metadata(metadata_path, metadata)

    return {
        "success": True,
        "counts": counts,
        "output_files": output_files,
        "metadata_file": metadata_path,
        "hours_scanned": scanned_hour_urls,
    }


def gh_archive_step(
    *,
    name: str = "raw/gh_archive/structured_output_eval",
    start_date: str,
    end_date: str,
    start_hour: int = 0,
    end_hour: int = 23,
    event_types: Sequence[str] = GH_ARCHIVE_DEFAULT_EVENT_TYPES,
    max_events_per_event_type: int | None = None,
    request_timeout: int = 120,
    base_url: str = GH_ARCHIVE_BASE_URL,
    metadata_filename: str = "metadata.json",
    skip_existing: bool = True,
    deps: list[StepSpec] | None = None,
    output_path_prefix: str | None = None,
    override_output_path: str | None = None,
) -> StepSpec:
    resolved_event_types = tuple(dict.fromkeys(event_types))

    def _run(output_path: str) -> dict[str, Any]:
        cfg = GhArchiveDownloadConfig(
            output_path=output_path,
            start_date=start_date,
            end_date=end_date,
            start_hour=start_hour,
            end_hour=end_hour,
            event_types=resolved_event_types,
            max_events_per_event_type=max_events_per_event_type,
            request_timeout=request_timeout,
            base_url=base_url,
            metadata_filename=metadata_filename,
            skip_existing=skip_existing,
        )
        return download_gh_archive_events(cfg)

    return StepSpec(
        name=name,
        fn=_run,
        deps=deps or [],
        hash_attrs={
            "start_date": start_date,
            "end_date": end_date,
            "start_hour": start_hour,
            "end_hour": end_hour,
            "event_types": resolved_event_types,
            "max_events_per_event_type": max_events_per_event_type,
            "base_url": base_url,
            "metadata_filename": metadata_filename,
        },
        output_path_prefix=output_path_prefix,
        override_output_path=override_output_path,
    )


def make_gh_archive_step(
    *,
    name: str = "raw/gh_archive/structured_output_eval",
    start_date: str,
    end_date: str,
    start_hour: int = 0,
    end_hour: int = 23,
    event_types: Sequence[str] = GH_ARCHIVE_DEFAULT_EVENT_TYPES,
    max_events_per_event_type: int | None = None,
    request_timeout: int = 120,
    base_url: str = GH_ARCHIVE_BASE_URL,
    metadata_filename: str = "metadata.json",
    skip_existing: bool = True,
) -> ExecutorStep:
    """Create an ExecutorStep that downloads and normalizes held-out GH Archive events."""
    return gh_archive_step(
        name=name,
        start_date=start_date,
        end_date=end_date,
        start_hour=start_hour,
        end_hour=end_hour,
        event_types=event_types,
        max_events_per_event_type=max_events_per_event_type,
        request_timeout=request_timeout,
        base_url=base_url,
        metadata_filename=metadata_filename,
        skip_existing=skip_existing,
    ).as_executor_step()
