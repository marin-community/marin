# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Download and normalize the latest Uncheatable Eval data dumps."""

from __future__ import annotations

import json
import logging
import os
import posixpath
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import fsspec
import requests
from marin.execution import THIS_OUTPUT_PATH, ExecutorStep, VersionedValue, ensure_versioned, this_output_path
from marin.utils import fsspec_mkdirs
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
from zephyr import Dataset, ZephyrContext
from zephyr.writers import atomic_rename

logger = logging.getLogger("ray")

FILENAME_PATTERN = re.compile(r"^(?P<benchmark>.+)_(?P<start>\d{8})to(?P<end>\d{8})(?P<suffix>(?:\.[^.]+)*)$")

TEXT_FIELD_CANDIDATES: tuple[str, ...] = (
    "text",
    "body",
    "content",
    "article",
    "document",
    "raw_text",
    "code",
    "message",
    "description",
    "story",
)

LIST_FIELD_CANDIDATES: tuple[str, ...] = (
    "paragraphs",
    "sentences",
    "lines",
    "messages",
)

ID_FIELD_CANDIDATES: tuple[str, ...] = (
    "id",
    "uuid",
    "guid",
    "doc_id",
    "document_id",
    "article_id",
    "hash",
    "sha",
    "uid",
)


@dataclass(frozen=True)
class UncheatableEvalDataset:
    """Information about a single data dump file from the Uncheatable Eval repository."""

    benchmark: str
    start_date: str
    end_date: str
    name: str
    download_url: str
    sha: str | None = None
    size: int | None = None

    @property
    def date_range(self) -> str:
        return f"{self.start_date}to{self.end_date}"

    @property
    def source_label(self) -> str:
        return f"{self.benchmark}:{self.date_range}"

    def output_filename(self, suffix: str = ".jsonl.gz") -> str:
        return f"{self.benchmark}_{self.date_range}{suffix}"


@dataclass
class UncheatableEvalDownloadConfig:
    """Configuration for downloading and normalizing Uncheatable Eval dumps."""

    output_path: str | VersionedValue[str] = THIS_OUTPUT_PATH
    repo_owner: str | VersionedValue[str] = "Jellyfish042"
    repo_name: str | VersionedValue[str] = "uncheatable_eval"
    data_path: str | VersionedValue[str] = "data"
    branch: str | VersionedValue[str] = "master"
    max_concurrent_downloads: int = 8
    request_timeout: int = 120
    github_token: str | None = None
    skip_existing: bool = True
    metadata_filename: str = "metadata.json"


def _http_headers(cfg: UncheatableEvalDownloadConfig) -> dict[str, str]:
    headers = {"Accept": "application/vnd.github+json"}
    token = cfg.github_token or os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _fetch_directory_listing(cfg: UncheatableEvalDownloadConfig) -> list[dict[str, Any]]:
    """Return the list of files in the configured GitHub repository directory."""

    headers = _http_headers(cfg)
    base_url = f"https://api.github.com/repos/{cfg.repo_owner!s}/{cfg.repo_name!s}/contents/{cfg.data_path!s}"
    params = {"ref": str(cfg.branch)}
    response = requests.get(base_url, headers=headers, params=params, timeout=cfg.request_timeout)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise ValueError(f"Unexpected response from GitHub API: {payload!r}")
    return payload


def _parse_available_dumps(entries: Iterable[dict[str, Any]]) -> list[UncheatableEvalDataset]:
    """Parse GitHub directory entries into dataset metadata."""

    datasets: list[UncheatableEvalDataset] = []
    for entry in entries:
        name = entry.get("name")
        if not isinstance(name, str):
            continue
        match = FILENAME_PATTERN.match(name)
        if not match:
            continue
        benchmark = match.group("benchmark")
        start = match.group("start")
        end = match.group("end")
        download_url = entry.get("download_url")
        if not isinstance(download_url, str):
            logger.debug("Skipping %s because it has no download_url", name)
            continue
        datasets.append(
            UncheatableEvalDataset(
                benchmark=benchmark,
                start_date=start,
                end_date=end,
                name=name,
                download_url=download_url,
                sha=entry.get("sha"),
                size=entry.get("size"),
            )
        )
    return datasets


def _select_latest_dumps(datasets: Iterable[UncheatableEvalDataset]) -> list[UncheatableEvalDataset]:
    """Select the latest dump for each benchmark based on the end date (and start date as tie breaker)."""

    latest: dict[str, UncheatableEvalDataset] = {}
    for dataset in datasets:
        existing = latest.get(dataset.benchmark)
        if existing is None:
            latest[dataset.benchmark] = dataset
            continue
        candidate_key = (dataset.end_date, dataset.start_date, dataset.name)
        existing_key = (existing.end_date, existing.start_date, existing.name)
        if candidate_key > existing_key:
            latest[dataset.benchmark] = dataset
    return sorted(latest.values(), key=lambda d: d.benchmark)


def _extract_id(raw: Any, dataset: UncheatableEvalDataset, index: int) -> str:
    if isinstance(raw, dict):
        for key in ID_FIELD_CANDIDATES:
            value = raw.get(key)
            if value:
                return str(value)
        metadata = raw.get("metadata")
        if isinstance(metadata, dict):
            for key in ID_FIELD_CANDIDATES:
                value = metadata.get(key)
                if value:
                    return str(value)
    return f"{dataset.benchmark}_{dataset.date_range}_{index:06d}"


def _join_list_field(value: Any) -> str | None:
    if isinstance(value, list):
        text_items = [str(item) for item in value if item is not None]
        if text_items:
            return "\n".join(text_items)
    return None


def _extract_text(raw: Any) -> str | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        return raw
    if isinstance(raw, dict):
        for key in TEXT_FIELD_CANDIDATES:
            value = raw.get(key)
            if isinstance(value, str) and value.strip():
                return value
        for key in TEXT_FIELD_CANDIDATES:
            value = raw.get(key)
            joined = _join_list_field(value)
            if joined:
                return joined
        for key in LIST_FIELD_CANDIDATES:
            joined = _join_list_field(raw.get(key))
            if joined:
                return joined
        title = raw.get("title")
        body = raw.get("body")
        if isinstance(title, str) and isinstance(body, str):
            combined = f"{title.strip()}\n\n{body.strip()}"
            if combined.strip():
                return combined
        if isinstance(title, str) and title.strip():
            return title
        return json.dumps(raw, ensure_ascii=False)
    return str(raw)


def _normalize_record(raw: Any, dataset: UncheatableEvalDataset, index: int) -> dict[str, str]:
    text = _extract_text(raw)
    if text is None or not str(text).strip():
        raise ValueError(f"Record {index} in {dataset.name} does not contain text")
    record_id = _extract_id(raw, dataset, index)
    return {"id": record_id, "text": text, "source": dataset.source_label}


def _download_and_convert_single(
    task: DownloadTask,
) -> dict[str, Any]:
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=1.0, status_forcelist=[500, 502, 503, 504], allowed_methods=["GET"])
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    logger.info("Downloading %s from %s", task.dataset.name, task.download_url)
    response = session.get(task.download_url, timeout=task.cfg.request_timeout, headers=_http_headers(task.cfg))
    response.raise_for_status()

    try:
        payload = response.json()
    except ValueError as exc:
        raise ValueError(f"Failed to decode JSON payload for {task.dataset.name}") from exc

    if not isinstance(payload, list):
        raise ValueError(f"Expected list in dataset {task.dataset.name}, found {type(payload).__name__}")

    fsspec_mkdirs(os.path.dirname(task.output_file_path), exist_ok=True)

    record_count = 0
    with atomic_rename(task.output_file_path) as temp_path:
        with fsspec.open(temp_path, "wt", encoding="utf-8", compression="gzip") as outfile:
            for index, raw in enumerate(payload):
                normalized = _normalize_record(raw, task.dataset, index)
                json.dump(normalized, outfile, ensure_ascii=False)
                outfile.write("\n")
                record_count += 1

    logger.info("Wrote %s records to %s", record_count, task.output_file_path)
    return {"records": record_count, "output_file": task.output_file_path}


@dataclass
class DownloadTask:
    download_url: str
    output_file_path: str
    dataset: UncheatableEvalDataset
    cfg: UncheatableEvalDownloadConfig


def _generate_tasks(
    datasets: Iterable[UncheatableEvalDataset],
    cfg: UncheatableEvalDownloadConfig,
) -> tuple[list[DownloadTask], list[UncheatableEvalDataset]]:
    tasks: list[DownloadTask] = []
    filtered: list[UncheatableEvalDataset] = []
    for dataset in datasets:
        output_file = posixpath.join(str(cfg.output_path), dataset.output_filename())
        tasks.append(DownloadTask(dataset.download_url, output_file, dataset, cfg))
        filtered.append(dataset)
    return tasks, filtered


def _write_metadata(cfg: UncheatableEvalDownloadConfig, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    metadata_path = posixpath.join(str(cfg.output_path), cfg.metadata_filename)
    with fsspec.open(metadata_path, "w", encoding="utf-8") as meta_file:
        json.dump(records, meta_file, indent=2, ensure_ascii=False)
    logger.info("Wrote metadata to %s", metadata_path)


def download_latest_uncheatable_eval(cfg: UncheatableEvalDownloadConfig) -> dict[str, Any]:
    """Download and normalize the newest Uncheatable Eval dump for each benchmark."""

    entries = _fetch_directory_listing(cfg)
    datasets = _parse_available_dumps(entries)
    latest_datasets = _select_latest_dumps(datasets)

    if not latest_datasets:
        logger.warning("No datasets found that match the expected naming pattern")
        return {"success": False, "reason": "no_datasets"}

    output_path = str(cfg.output_path)
    fsspec_mkdirs(output_path, exist_ok=True)

    tasks, filtered_datasets = _generate_tasks(latest_datasets, cfg)

    if not tasks:
        logger.info("No new datasets to process")
        return {"success": True, "reason": "already_processed", "skipped": True}

    metadata_records: list[dict[str, Any]] = []

    pipeline = (
        Dataset.from_list(tasks)
        .map(lambda task: _download_and_convert_single(task))
        .write_jsonl(f"{cfg.output_path}/.metrics/part-{{shard:05d}}.jsonl", skip_existing=True)
    )
    with ZephyrContext(name="download-uncheatable-eval") as ctx:
        output_paths = ctx.execute(pipeline)

    for dataset, metadata_file in zip(filtered_datasets, output_paths, strict=True):
        with fsspec.open(metadata_file, "r", encoding="utf-8") as meta_file:
            result = json.load(meta_file)

        try:
            metadata_records.append(
                {
                    "benchmark": dataset.benchmark,
                    "start_date": dataset.start_date,
                    "end_date": dataset.end_date,
                    "source": dataset.source_label,
                    "output_file": posixpath.join(output_path, dataset.output_filename()),
                    "records": result.get("records"),
                    "sha": dataset.sha,
                    "size": dataset.size,
                }
            )
        except Exception:
            logger.exception("Failed to process dataset %s", dataset.name)
            raise

    _write_metadata(cfg, metadata_records)
    return {"success": True, "processed": metadata_records}


def make_uncheatable_eval_step(
    *,
    name: str = "raw/uncheatable-eval/latest",
    repo_owner: str = "ziqing-huang",
    repo_name: str = "uncheatable_eval",
    data_path: str = "data",
    branch: str = "master",
    max_concurrent_downloads: int = 8,
    request_timeout: int = 120,
    github_token: str | None = None,
    skip_existing: bool = True,
) -> ExecutorStep[UncheatableEvalDownloadConfig]:
    """Create an :class:`ExecutorStep` that downloads the latest Uncheatable Eval dumps."""

    config = UncheatableEvalDownloadConfig(
        output_path=this_output_path(),
        repo_owner=ensure_versioned(repo_owner),
        repo_name=ensure_versioned(repo_name),
        data_path=ensure_versioned(data_path),
        branch=ensure_versioned(branch),
        max_concurrent_downloads=max_concurrent_downloads,
        request_timeout=request_timeout,
        github_token=github_token,
        skip_existing=skip_existing,
    )

    return ExecutorStep(
        name=name,
        fn=download_latest_uncheatable_eval,
        config=config,
    )


__all__ = [
    "UncheatableEvalDataset",
    "UncheatableEvalDownloadConfig",
    "download_latest_uncheatable_eval",
    "make_uncheatable_eval_step",
]
