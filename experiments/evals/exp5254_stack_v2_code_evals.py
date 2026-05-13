# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Materialize Stack v2 per-language held-out raw eval slices for issue #5254."""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import posixpath
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime
from functools import partial
from typing import Any

import requests
from datasets import load_dataset
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from marin.utils import fsspec_mkdirs
from requests.adapters import HTTPAdapter
from rigging.filesystem import open_url
from urllib3.util import Retry
from zephyr.writers import atomic_rename

from experiments.evals.long_tail_ppl import (
    CODE_ECOSYSTEM_LANGUAGES,
    CODE_ECOSYSTEM_LARGE_TARGET_TOKENS,
    CODE_ECOSYSTEM_SMALL_TARGET_TOKENS,
    STACK_V2_DATASET_ID,
    STACK_V2_REVISION,
    CodeEcosystemTier,
    _language_to_slug,
    stack_v2_config_name,
)

logger = logging.getLogger(__name__)

OUTPUT_FILENAME = "heldout.jsonl.gz"
METADATA_FILENAME = "heldout_metadata.json"
SWH_CONTENT_URL_TEMPLATE = "https://archive.softwareheritage.org/api/1/content/sha1:{blob_id}/raw/"
MIN_LENGTH_BYTES = 256
MAX_LENGTH_BYTES = 2_000_000
HTTP_TIMEOUT = 60
REQUEST_SLEEP = 0.2


@dataclass(frozen=True)
class StackV2HeldoutConfig:
    """Configuration for one Stack v2 held-out raw eval slice."""

    language: str
    stack_v2_config: str
    target_compressed_bytes: int
    dataset_id: str = STACK_V2_DATASET_ID
    revision: str = STACK_V2_REVISION
    min_length_bytes: int = MIN_LENGTH_BYTES
    max_length_bytes: int = MAX_LENGTH_BYTES
    output_filename: str = OUTPUT_FILENAME
    request_sleep: float = REQUEST_SLEEP
    http_timeout: int = HTTP_TIMEOUT


def _json_default(value: Any) -> str:
    if isinstance(value, datetime | date):
        return value.isoformat()
    return str(value)


def _build_session() -> requests.Session:
    retry = Retry(
        total=8,
        connect=5,
        read=5,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def _download_swh_content(session: requests.Session, blob_id: str, src_encoding: str | None, timeout: int) -> str:
    url = SWH_CONTENT_URL_TEMPLATE.format(blob_id=blob_id)
    response = session.get(url, timeout=timeout)
    response.raise_for_status()
    encoding = src_encoding or "utf-8"
    return response.content.decode(encoding)


def _record_from_row(
    row: dict[str, Any],
    text: str,
    language: str,
    index: int,
    config: StackV2HeldoutConfig,
) -> dict[str, Any]:
    return {
        "id": f"stack_v2:{_language_to_slug(language)}:{index:08d}:{row['blob_id']}",
        "text": text,
        "source": "bigcode/the-stack-v2",
        "language": language,
        "provenance": {
            "dataset_id": config.dataset_id,
            "revision": config.revision,
            "stack_v2_config": config.stack_v2_config,
            "metadata_index": index,
            "blob_id": row["blob_id"],
            "content_id": row.get("content_id"),
            "repo_name": row.get("repo_name"),
            "path": row.get("path"),
            "revision_id": row.get("revision_id"),
            "license_type": row.get("license_type"),
            "detected_licenses": row.get("detected_licenses"),
            "src_encoding": row.get("src_encoding"),
            "length_bytes": row.get("length_bytes"),
            "is_vendor": row.get("is_vendor"),
            "is_generated": row.get("is_generated"),
        },
    }


def _should_consider_row(row: dict[str, Any], config: StackV2HeldoutConfig) -> bool:
    length_bytes = row.get("length_bytes")
    if not isinstance(length_bytes, int):
        return False
    if length_bytes < config.min_length_bytes or length_bytes > config.max_length_bytes:
        return False
    return not bool(row.get("is_vendor")) and not bool(row.get("is_generated"))


def materialize_stack_v2_heldout(config: StackV2HeldoutConfig, output_path: str) -> dict[str, Any]:
    """Write one bounded Stack v2 held-out slice as raw-text JSONL.gz."""

    fsspec_mkdirs(output_path, exist_ok=True)
    output_file = posixpath.join(output_path, config.output_filename)
    session = _build_session()
    record_count = 0
    text_bytes = 0
    skipped_decode_errors = 0
    skipped_download_errors = 0
    rows_seen = 0

    try:
        with tempfile.TemporaryDirectory(prefix="stack-v2-heldout-") as temp_dir:
            local_gzip = os.path.join(temp_dir, config.output_filename)
            with gzip.open(local_gzip, "wt", encoding="utf-8") as handle:
                dataset = load_dataset(
                    config.dataset_id,
                    name=config.stack_v2_config,
                    split="train",
                    revision=config.revision,
                    streaming=True,
                )
                for index, row in enumerate(dataset):
                    rows_seen = index + 1
                    if not _should_consider_row(row, config):
                        continue
                    try:
                        text = _download_swh_content(
                            session,
                            str(row["blob_id"]),
                            row.get("src_encoding"),
                            config.http_timeout,
                        )
                    except (LookupError, UnicodeError):
                        skipped_decode_errors += 1
                        continue
                    except requests.RequestException:
                        skipped_download_errors += 1
                        continue

                    if not text.strip():
                        continue
                    record = _record_from_row(row, text, config.language, index, config)
                    json.dump(record, handle, ensure_ascii=False, sort_keys=True, default=_json_default)
                    handle.write("\n")
                    record_count += 1
                    text_bytes += len(text.encode("utf-8"))
                    handle.flush()

                    compressed_bytes = os.path.getsize(local_gzip)
                    if compressed_bytes >= config.target_compressed_bytes:
                        break
                    if config.request_sleep > 0:
                        time.sleep(config.request_sleep)

            compressed_bytes = os.path.getsize(local_gzip)
            if compressed_bytes < config.target_compressed_bytes:
                raise ValueError(
                    f"{config.language} exhausted before target: "
                    f"{compressed_bytes:,}/{config.target_compressed_bytes:,} compressed bytes, "
                    f"{record_count} records, {rows_seen} metadata rows"
                )

            with atomic_rename(output_file) as temp_path:
                with open_url(temp_path, "wb") as dest, open(local_gzip, "rb") as src:
                    shutil.copyfileobj(src, dest)
    finally:
        session.close()

    metadata = {
        "config": asdict(config),
        "output_file": output_file,
        "record_count": record_count,
        "text_bytes": text_bytes,
        "compressed_bytes": compressed_bytes,
        "rows_seen": rows_seen,
        "skipped_decode_errors": skipped_decode_errors,
        "skipped_download_errors": skipped_download_errors,
    }
    metadata_file = posixpath.join(output_path, METADATA_FILENAME)
    with open_url(metadata_file, "w") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")

    logger.info(
        "Materialized %s: %d records, %d compressed bytes, %d text bytes",
        config.language,
        record_count,
        compressed_bytes,
        text_bytes,
    )
    return metadata


def stack_v2_heldout_steps(*, only_languages: set[str] | None = None) -> list[StepSpec]:
    """Return one materialization step per registered Stack v2 code language."""

    steps: list[StepSpec] = []
    for tier in CodeEcosystemTier:
        target = (
            CODE_ECOSYSTEM_LARGE_TARGET_TOKENS if tier == CodeEcosystemTier.LARGE else CODE_ECOSYSTEM_SMALL_TARGET_TOKENS
        )
        for language in CODE_ECOSYSTEM_LANGUAGES[tier]:
            if only_languages is not None and language not in only_languages:
                continue
            config = StackV2HeldoutConfig(
                language=language,
                stack_v2_config=stack_v2_config_name(language),
                target_compressed_bytes=target,
            )
            slug = _language_to_slug(language)
            steps.append(
                StepSpec(
                    name=f"evaluation/long_tail_ppl/stack_v2/{slug}",
                    fn=partial(materialize_stack_v2_heldout, config),
                    hash_attrs=asdict(config),
                    override_output_path=f"raw/long_tail_ppl/code/stack_v2/{slug}",
                )
            )
    return steps


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-concurrent", type=int, default=4)
    parser.add_argument("--language", action="append", help="Limit to one display language. Repeatable.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    only_languages = set(args.language) if args.language else None
    steps = stack_v2_heldout_steps(only_languages=only_languages)
    if not steps:
        raise ValueError("No Stack v2 held-out steps selected")
    StepRunner().run(steps, dry_run=args.dry_run, max_concurrent=args.max_concurrent)
