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
from dataclasses import asdict, dataclass
from datetime import date, datetime
from typing import Any

from datasets import load_dataset
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from marin.utils import fsspec_mkdirs
from rigging.filesystem import open_url, url_to_fs
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

COMMON_PILE_STACK_V2_DATASET_ID = "common-pile/stackv2"
COMMON_PILE_STACK_V2_REVISION = "d0e3266fce12d25de28f2576ffb7272c18b0148f"
OUTPUT_FILENAME = "heldout.jsonl.gz"
METADATA_FILENAME = "heldout_metadata.json"
SUMMARY_FILENAME = "materialization_summary.json"
MIN_LENGTH_BYTES = 256
MAX_LENGTH_BYTES = 2_000_000
PROGRESS_INTERVAL_ROWS = 50_000


@dataclass(frozen=True)
class StackV2HeldoutConfig:
    """Configuration for one Stack v2 held-out raw eval slice."""

    language: str
    stack_v2_config: str
    target_compressed_bytes: int
    source_dataset_id: str = COMMON_PILE_STACK_V2_DATASET_ID
    source_revision: str = COMMON_PILE_STACK_V2_REVISION
    stack_v2_dataset_id: str = STACK_V2_DATASET_ID
    stack_v2_revision: str = STACK_V2_REVISION
    min_length_bytes: int = MIN_LENGTH_BYTES
    max_length_bytes: int = MAX_LENGTH_BYTES
    output_filename: str = OUTPUT_FILENAME


def _json_default(value: Any) -> str:
    if isinstance(value, datetime | date):
        return value.isoformat()
    return str(value)


def _path_exists(path: str) -> bool:
    fs, resolved_path = url_to_fs(path)
    return fs.exists(resolved_path)


def _target_configs(*, only_languages: set[str] | None = None) -> dict[str, StackV2HeldoutConfig]:
    configs: dict[str, StackV2HeldoutConfig] = {}
    for tier in CodeEcosystemTier:
        target = (
            CODE_ECOSYSTEM_LARGE_TARGET_TOKENS if tier == CodeEcosystemTier.LARGE else CODE_ECOSYSTEM_SMALL_TARGET_TOKENS
        )
        for language in CODE_ECOSYSTEM_LANGUAGES[tier]:
            if only_languages is not None and language not in only_languages:
                continue
            configs[language] = StackV2HeldoutConfig(
                language=language,
                stack_v2_config=stack_v2_config_name(language),
                target_compressed_bytes=target,
            )
    return configs


def _record_from_row(row: dict[str, Any], language: str, index: int, config: StackV2HeldoutConfig) -> dict[str, Any]:
    metadata = row.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    row_id = row.get("id") or metadata.get("blob_id") or f"row-{index}"
    return {
        "id": f"stack_v2:{_language_to_slug(language)}:{index:08d}:{row_id}",
        "text": row["text"],
        "source": COMMON_PILE_STACK_V2_DATASET_ID,
        "language": language,
        "provenance": {
            "dataset_id": config.source_dataset_id,
            "revision": config.source_revision,
            "stack_v2_dataset_id": config.stack_v2_dataset_id,
            "stack_v2_revision": config.stack_v2_revision,
            "stack_v2_config": config.stack_v2_config,
            "metadata_index": index,
            "row_id": row.get("id"),
            "blob_id": metadata.get("blob_id"),
            "content_id": metadata.get("content_id"),
            "repo_name": metadata.get("repo_name"),
            "path": metadata.get("path"),
            "url": metadata.get("url"),
            "revision_id": metadata.get("revision_id"),
            "license_type": metadata.get("license_type"),
            "detected_licenses": metadata.get("detected_licenses"),
            "length_bytes": metadata.get("length_bytes"),
            "is_vendor": metadata.get("is_vendor"),
            "is_generated": metadata.get("is_generated"),
        },
    }


def _metadata_language(row: dict[str, Any]) -> str | None:
    metadata = row.get("metadata")
    if not isinstance(metadata, dict):
        return None
    language = metadata.get("language")
    return language if isinstance(language, str) else None


def _length_bytes(row: dict[str, Any]) -> int:
    text = row["text"]
    assert isinstance(text, str)
    metadata = row.get("metadata")
    if isinstance(metadata, dict) and isinstance(metadata.get("length_bytes"), int):
        return int(metadata["length_bytes"])
    return len(text.encode("utf-8"))


def _should_consider_row(row: dict[str, Any], config: StackV2HeldoutConfig) -> bool:
    if not isinstance(row.get("text"), str) or not row["text"].strip():
        return False
    length_bytes = _length_bytes(row)
    if length_bytes < config.min_length_bytes or length_bytes > config.max_length_bytes:
        return False
    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        return not bool(metadata.get("is_vendor")) and not bool(metadata.get("is_generated"))
    return True


def _language_output_path(base_output_path: str, language: str) -> str:
    return posixpath.join(base_output_path, _language_to_slug(language))


def _output_file(base_output_path: str, language: str, config: StackV2HeldoutConfig) -> str:
    return posixpath.join(_language_output_path(base_output_path, language), config.output_filename)


def _metadata_file(base_output_path: str, language: str) -> str:
    return posixpath.join(_language_output_path(base_output_path, language), METADATA_FILENAME)


def _existing_language_metadata(
    base_output_path: str, language: str, config: StackV2HeldoutConfig
) -> dict[str, Any] | None:
    output_file = _output_file(base_output_path, language, config)
    metadata_file = _metadata_file(base_output_path, language)
    if not _path_exists(output_file) or not _path_exists(metadata_file):
        return None
    with open_url(metadata_file, "r") as handle:
        metadata = json.load(handle)
    metadata["output_file"] = output_file
    return metadata


def _write_language_outputs(
    base_output_path: str,
    language: str,
    config: StackV2HeldoutConfig,
    local_file: str,
    language_stats: dict[str, Any],
) -> None:
    language_output_path = _language_output_path(base_output_path, language)
    fsspec_mkdirs(language_output_path, exist_ok=True)

    output_file = _output_file(base_output_path, language, config)
    with atomic_rename(output_file) as temp_path:
        with open_url(temp_path, "wb") as dest, open(local_file, "rb") as src:
            shutil.copyfileobj(src, dest)

    language_stats["output_file"] = output_file
    metadata_file = _metadata_file(base_output_path, language)
    with open_url(metadata_file, "w") as handle:
        json.dump(language_stats, handle, indent=2, sort_keys=True, default=_json_default)
        handle.write("\n")


def materialize_stack_v2_heldouts(
    output_path: str,
    *,
    only_languages: set[str] | None = None,
) -> dict[str, Any]:
    """Write all selected Stack v2 held-out slices as raw-text JSONL.gz."""

    selected = _target_configs(only_languages=only_languages)
    if not selected:
        raise ValueError("No Stack v2 held-out languages selected")

    base_output_path = posixpath.dirname(output_path.rstrip("/"))
    fsspec_mkdirs(base_output_path, exist_ok=True)
    completed: set[str] = set()
    rows_seen = 0
    skipped_rows = 0
    stats: dict[str, dict[str, Any]] = {}
    pending: dict[str, StackV2HeldoutConfig] = {}
    for language, config in selected.items():
        existing_metadata = _existing_language_metadata(base_output_path, language, config)
        if existing_metadata is not None:
            completed.add(language)
            stats[language] = existing_metadata
            logger.info("Skipping existing held-out slice for %s at %s", language, existing_metadata["output_file"])
            continue
        pending[language] = config

    with tempfile.TemporaryDirectory(prefix="stack-v2-heldouts-") as temp_dir:
        handles: dict[str, gzip.GzipFile] = {}
        local_files: dict[str, str] = {}
        for language, config in pending.items():
            slug = _language_to_slug(language)
            local_file = os.path.join(temp_dir, slug, OUTPUT_FILENAME)
            os.makedirs(os.path.dirname(local_file), exist_ok=True)
            local_files[language] = local_file
            handles[language] = gzip.open(local_file, "wt", encoding="utf-8")
            stats[language] = {
                "config": asdict(config),
                "record_count": 0,
                "text_bytes": 0,
                "compressed_bytes": 0,
                "first_row_index": None,
                "last_row_index": None,
            }

        try:
            if not pending:
                logger.info("All %d selected held-out slices already exist", len(selected))
            else:
                dataset = load_dataset(
                    COMMON_PILE_STACK_V2_DATASET_ID,
                    split="train",
                    revision=COMMON_PILE_STACK_V2_REVISION,
                    streaming=True,
                )
                for index, row in enumerate(dataset):
                    rows_seen = index + 1
                    language = _metadata_language(row)
                    if language not in pending or language in completed:
                        skipped_rows += 1
                        continue
                    config = pending[language]
                    if not _should_consider_row(row, config):
                        skipped_rows += 1
                        continue

                    record = _record_from_row(row, language, index, config)
                    handle = handles[language]
                    json.dump(record, handle, ensure_ascii=False, sort_keys=True, default=_json_default)
                    handle.write("\n")
                    handle.flush()

                    text_bytes = len(row["text"].encode("utf-8"))
                    language_stats = stats[language]
                    language_stats["record_count"] += 1
                    language_stats["text_bytes"] += text_bytes
                    language_stats["last_row_index"] = index
                    if language_stats["first_row_index"] is None:
                        language_stats["first_row_index"] = index

                    compressed_bytes = os.path.getsize(local_files[language])
                    language_stats["compressed_bytes"] = compressed_bytes
                    if compressed_bytes >= config.target_compressed_bytes:
                        handle.close()
                        del handles[language]
                        _write_language_outputs(
                            base_output_path,
                            language,
                            config,
                            local_files[language],
                            language_stats,
                        )
                        completed.add(language)
                        logger.info(
                            "Completed and uploaded %s: %d records, %d compressed bytes after %d rows",
                            language,
                            language_stats["record_count"],
                            compressed_bytes,
                            rows_seen,
                        )

                    if rows_seen % PROGRESS_INTERVAL_ROWS == 0:
                        logger.info(
                            "Scanned %d rows; completed %d/%d slices",
                            rows_seen,
                            len(completed),
                            len(selected),
                        )
                    if len(completed) == len(selected):
                        break
        finally:
            for handle in handles.values():
                handle.close()

        missing = sorted(set(selected) - completed)
        if missing:
            raise ValueError(f"Missing held-out targets after scanning {rows_seen:,} rows: {missing}")

    summary = {
        "source_dataset_id": COMMON_PILE_STACK_V2_DATASET_ID,
        "source_revision": COMMON_PILE_STACK_V2_REVISION,
        "stack_v2_dataset_id": STACK_V2_DATASET_ID,
        "stack_v2_revision": STACK_V2_REVISION,
        "rows_seen": rows_seen,
        "skipped_rows": skipped_rows,
        "completed_languages": sorted(completed),
        "slice_count": len(completed),
        "stats": stats,
    }
    summary_file = posixpath.join(output_path, SUMMARY_FILENAME)
    fsspec_mkdirs(output_path, exist_ok=True)
    with open_url(summary_file, "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, default=_json_default)
        handle.write("\n")
    return summary


def stack_v2_heldout_steps(*, only_languages: set[str] | None = None) -> list[StepSpec]:
    """Return the Stack v2 code held-out materialization step."""

    selected = _target_configs(only_languages=only_languages)
    return [
        StepSpec(
            name="evaluation/long_tail_ppl/stack_v2/materialize_all",
            fn=lambda output_path: materialize_stack_v2_heldouts(output_path, only_languages=set(selected)),
            hash_attrs={
                "source_dataset_id": COMMON_PILE_STACK_V2_DATASET_ID,
                "source_revision": COMMON_PILE_STACK_V2_REVISION,
                "selected": {language: asdict(config) for language, config in selected.items()},
            },
            override_output_path="raw/long_tail_ppl/code/stack_v2/_materialize_all",
        )
    ]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-concurrent", type=int, default=1)
    parser.add_argument("--language", action="append", help="Limit to one display language. Repeatable.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    only_languages = set(args.language) if args.language else None
    steps = stack_v2_heldout_steps(only_languages=only_languages)
    StepRunner().run(steps, dry_run=args.dry_run, max_concurrent=args.max_concurrent)
