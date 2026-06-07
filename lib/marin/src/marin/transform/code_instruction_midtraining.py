# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Transform code-instruction Hugging Face datasets into Dolma LM documents."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import datasets
import draccus
from marin.execution import unwrap_versioned_value
from marin.utils import fsspec_mkdirs, load_dataset_with_backoff
from rigging.filesystem import url_to_fs
from zephyr import Dataset, ZephyrContext, load_jsonl, write_jsonl_file

logger = logging.getLogger(__name__)

DEFAULT_SAMPLE_SEED = "marin-opencode-midtraining-v1"


@dataclass(frozen=True)
class CodeInstructionMidtrainingConfig:
    """Configuration for code-instruction datasets staged as Dolma documents.

    Args:
        source: Hugging Face dataset id.
        revision: Pinned Hugging Face revision.
        output_path: Destination directory for transformed JSONL shards.
        instruction_column: Source column containing the problem or prompt.
        output_column: Source column containing the target response.
        solution_column: Optional source column containing a reference solution.
        metadata_columns: Source row columns to copy into the Dolma document metadata.
        subsets: Dataset configs to process. Empty means all available configs.
        splits: Dataset splits to process. Empty means all available splits.
        average_test_score_column: Optional column used for quality filtering.
        min_average_test_score: Minimum inclusive score for rows with an average test score.
        allowed_domains: Optional allow-list for the ``domain`` column.
        allowed_generation_algorithms: Optional allow-list for ``generation_algorithm``.
        allowed_generation_models: Optional allow-list for ``generation_model``.
        allowed_last_operations: Optional allow-list for ``last_operation``.
        sample_fraction: Deterministic row sample fraction after quality filtering.
        sample_seed: Stable seed for deterministic sampling.
        max_text_chars: Drop rendered documents above this character count.
        max_parallelism: Maximum Zephyr shard workers.
    """

    source: str
    revision: str
    output_path: str
    instruction_column: str
    output_column: str
    solution_column: str | None = None
    metadata_columns: list[str] = field(default_factory=list)
    subsets: list[str] = field(default_factory=list)
    splits: list[str] = field(default_factory=lambda: ["train"])
    average_test_score_column: str | None = None
    min_average_test_score: float | None = None
    allowed_domains: list[str] = field(default_factory=list)
    allowed_generation_algorithms: list[str] = field(default_factory=list)
    allowed_generation_models: list[str] = field(default_factory=list)
    allowed_last_operations: list[str] = field(default_factory=list)
    sample_fraction: float = 1.0
    sample_seed: str = DEFAULT_SAMPLE_SEED
    max_text_chars: int | None = 200_000
    max_parallelism: int | None = 32


@dataclass(frozen=True)
class CodeInstructionShardTask:
    """Task for processing one streaming Hugging Face shard."""

    source: str
    revision: str
    subset: str | None
    split: str
    shard_idx: int
    num_shards: int
    output_path: str
    cfg: CodeInstructionMidtrainingConfig


@dataclass(frozen=True)
class RenderedCodeInstruction:
    """A rendered code-instruction row before Dolma document wrapping."""

    row_id: str
    text: str
    metadata: dict[str, Any]


def _string_value(row: dict[str, Any], column: str) -> str | None:
    value = row.get(column)
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def _allowed_value(row: dict[str, Any], column: str, allowed_values: Sequence[str], counters: Counter[str]) -> bool:
    if not allowed_values:
        return True

    value = row.get(column)
    if value is None or str(value) not in allowed_values:
        counters[f"dropped_{column}"] += 1
        return False

    return True


def _parse_score(row: dict[str, Any], score_column: str, counters: Counter[str]) -> float | None:
    value = row.get(score_column)
    try:
        score = float(value)
    except (TypeError, ValueError):
        counters["dropped_missing_score"] += 1
        return None
    if not math.isfinite(score):
        counters["dropped_missing_score"] += 1
        return None
    return score


def _hash_fraction(key: str) -> float:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    numerator = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return numerator / 2**64


def _content_identity(instruction: str, output: str, solution: str | None) -> str:
    payload = {"instruction": instruction, "output": output, "solution": solution or ""}
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _row_id(row: dict[str, Any], content_identity: str) -> str:
    value = row.get("id")
    if value is not None:
        row_id = str(value).strip()
        if row_id:
            return row_id
    return hashlib.sha256(content_identity.encode("utf-8")).hexdigest()


def _sample_key(row_id: str, cfg: CodeInstructionMidtrainingConfig, subset: str, split: str) -> str:
    payload = {
        "sample_seed": unwrap_versioned_value(cfg.sample_seed),
        "source": unwrap_versioned_value(cfg.source),
        "revision": unwrap_versioned_value(cfg.revision),
        "subset": subset,
        "split": split,
        "row_id": row_id,
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _keep_sample(row_id: str, cfg: CodeInstructionMidtrainingConfig, subset: str, split: str) -> bool:
    sample_fraction = unwrap_versioned_value(cfg.sample_fraction)
    if not math.isfinite(sample_fraction) or sample_fraction < 0 or sample_fraction > 1:
        raise ValueError(f"sample_fraction must be between 0 and 1, got {sample_fraction}.")
    if sample_fraction >= 1:
        return True
    return _hash_fraction(_sample_key(row_id, cfg, subset, split)) < sample_fraction


def _render_text(instruction: str, output: str, solution: str | None) -> str:
    parts = [f"Problem:\n{instruction}"]
    if solution is not None:
        parts.append(f"Reference solution:\n{solution}")
    parts.append(f"Answer:\n{output}")
    return "\n\n".join(parts)


def _metadata(row: dict[str, Any], cfg: CodeInstructionMidtrainingConfig, subset: str, split: str) -> dict[str, Any]:
    metadata = {
        "hf_dataset_id": unwrap_versioned_value(cfg.source),
        "hf_revision": unwrap_versioned_value(cfg.revision),
        "hf_subset": subset,
        "hf_split": split,
    }

    row_id = row.get("id")
    if row_id is not None:
        metadata["row_id"] = row_id

    for column in unwrap_versioned_value(cfg.metadata_columns):
        if column in row:
            metadata[column] = row[column]

    return metadata


def render_code_instruction_row(
    row: dict[str, Any],
    cfg: CodeInstructionMidtrainingConfig,
    *,
    subset: str,
    split: str,
    counters: Counter[str] | None = None,
) -> RenderedCodeInstruction | None:
    """Render a source row as code-instruction LM text, returning ``None`` for filtered rows."""

    counters = counters if counters is not None else Counter()
    instruction = _string_value(row, unwrap_versioned_value(cfg.instruction_column))
    if instruction is None:
        counters["dropped_missing_instruction"] += 1
        return None

    output = _string_value(row, unwrap_versioned_value(cfg.output_column))
    if output is None:
        counters["dropped_missing_output"] += 1
        return None

    solution_column = unwrap_versioned_value(cfg.solution_column)
    solution = None
    if solution_column is not None:
        solution = _string_value(row, solution_column)
        if solution is None:
            counters["dropped_missing_solution"] += 1
            return None

    score_column = unwrap_versioned_value(cfg.average_test_score_column)
    min_score = unwrap_versioned_value(cfg.min_average_test_score)
    if score_column is not None and min_score is not None:
        score = _parse_score(row, score_column, counters)
        if score is None:
            return None
        if score < min_score:
            counters["dropped_below_score"] += 1
            return None

    if not _allowed_value(row, "domain", unwrap_versioned_value(cfg.allowed_domains), counters):
        return None
    if not _allowed_value(
        row, "generation_algorithm", unwrap_versioned_value(cfg.allowed_generation_algorithms), counters
    ):
        return None
    if not _allowed_value(row, "generation_model", unwrap_versioned_value(cfg.allowed_generation_models), counters):
        return None
    if not _allowed_value(row, "last_operation", unwrap_versioned_value(cfg.allowed_last_operations), counters):
        return None

    content_identity = _content_identity(instruction, output, solution)
    row_id = _row_id(row, content_identity)
    if not _keep_sample(row_id, cfg, subset, split):
        counters["dropped_sample"] += 1
        return None

    text = _render_text(instruction, output, solution)
    max_text_chars = unwrap_versioned_value(cfg.max_text_chars)
    if max_text_chars is not None and len(text) > max_text_chars:
        counters["dropped_too_long"] += 1
        return None

    return RenderedCodeInstruction(
        row_id=row_id,
        text=text,
        metadata=_metadata(row, cfg, subset=subset, split=split),
    )


def _document_id(
    rendered: RenderedCodeInstruction,
    cfg: CodeInstructionMidtrainingConfig,
    subset: str,
    split: str,
) -> str:
    payload = {
        "source": unwrap_versioned_value(cfg.source),
        "revision": unwrap_versioned_value(cfg.revision),
        "subset": subset,
        "split": split,
        "row_id": rendered.row_id,
        "text": rendered.text,
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def iter_code_instruction_documents(
    rows: Iterable[dict[str, Any]],
    cfg: CodeInstructionMidtrainingConfig,
    *,
    subset: str,
    split: str,
    counters: Counter[str] | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield Dolma documents from rows while collecting filter and dedup counters."""

    counters = counters if counters is not None else Counter()
    seen_text_hashes: set[bytes] = set()
    source = unwrap_versioned_value(cfg.source)
    added = datetime.now(UTC).isoformat()

    for row in rows:
        counters["input_rows"] += 1
        rendered = render_code_instruction_row(row, cfg, subset=subset, split=split, counters=counters)
        if rendered is None:
            continue

        text_hash = hashlib.sha256(rendered.text.strip().encode("utf-8")).digest()
        if text_hash in seen_text_hashes:
            counters["dropped_duplicate_text"] += 1
            continue
        seen_text_hashes.add(text_hash)

        counters["output_rows"] += 1
        yield {
            "id": _document_id(rendered, cfg, subset, split),
            "text": rendered.text,
            "source": source,
            "added": added,
            "metadata": rendered.metadata,
        }


def _get_available_subsets(cfg: CodeInstructionMidtrainingConfig) -> list[str | None]:
    source = unwrap_versioned_value(cfg.source)
    revision = unwrap_versioned_value(cfg.revision)
    configured_subsets = unwrap_versioned_value(cfg.subsets)
    if configured_subsets:
        selected_subsets: list[str | None] = list(configured_subsets)
        return selected_subsets

    available = datasets.get_dataset_config_names(source, revision=revision)
    return [None if name == "default" else name for name in available]


def _get_available_splits(cfg: CodeInstructionMidtrainingConfig, subset: str | None) -> list[str]:
    source = unwrap_versioned_value(cfg.source)
    revision = unwrap_versioned_value(cfg.revision)
    if subset not in (None, "default"):
        return datasets.get_dataset_split_names(source, name=subset, revision=revision)
    return datasets.get_dataset_split_names(source, revision=revision)


def _shard_filename(output_path: str, shard_idx: int) -> str:
    return os.path.join(output_path, f"{shard_idx:05d}.jsonl.gz")


def _shard_dir(output_path: str, subset: str, split: str) -> str:
    return os.path.join(output_path, subset, split)


def _create_output_directory(output_path: str) -> str:
    fsspec_mkdirs(output_path, exist_ok=True)
    return output_path


def get_code_instruction_shard_tasks(cfg: CodeInstructionMidtrainingConfig) -> Iterator[CodeInstructionShardTask]:
    """Enumerate streaming shard tasks for every configured subset and split."""

    source = unwrap_versioned_value(cfg.source)
    if not source:
        raise ValueError("Code-instruction transform requires a Hugging Face dataset id.")
    revision = unwrap_versioned_value(cfg.revision)
    configured_splits = unwrap_versioned_value(cfg.splits)

    for subset in _get_available_subsets(cfg):
        available_splits = _get_available_splits(cfg, subset)
        splits = available_splits
        if configured_splits:
            requested = set(configured_splits)
            missing = sorted(requested - set(available_splits))
            if missing:
                logger.warning("Requested split(s) %s for %s subset=%s skipped.", missing, source, subset)
            splits = [split for split in available_splits if split in requested]
        if not splits:
            logger.warning("No splits to process for %s subset=%s; skipping.", source, subset)
            continue

        for split in splits:
            subset_name = subset or "default"
            output_path = _create_output_directory(_shard_dir(cfg.output_path, subset_name, split))
            dataset_kwargs: dict[str, Any] = {
                "path": source,
                "split": split,
                "streaming": True,
                "revision": revision,
            }
            if subset not in (None, "default"):
                dataset_kwargs["name"] = subset
            dataset = load_dataset_with_backoff(
                context=f"{source} subset={subset_name} split={split}",
                **dataset_kwargs,
            )
            num_shards = dataset.num_shards
            if not num_shards:
                raise ValueError(f"Streaming dataset {source} subset={subset_name} split={split} lacks num_shards.")

            for shard_idx in range(num_shards):
                yield CodeInstructionShardTask(
                    source=source,
                    revision=revision,
                    subset=subset,
                    split=split,
                    shard_idx=shard_idx,
                    num_shards=num_shards,
                    output_path=output_path,
                    cfg=cfg,
                )


def process_code_instruction_shard_task(task: CodeInstructionShardTask) -> dict[str, Any]:
    """Transform and write one Hugging Face streaming shard."""

    subset_name = task.subset or "default"
    output_filename = _shard_filename(task.output_path, task.shard_idx)
    fs, _ = url_to_fs(output_filename)
    if fs.exists(output_filename):
        logger.info(
            "Skipping subset=%s split=%s shard=%s because output exists: %s",
            subset_name,
            task.split,
            task.shard_idx,
            output_filename,
        )
        return {
            "subset": subset_name,
            "split": task.split,
            "shard_idx": task.shard_idx,
            "path": output_filename,
            "count": 0,
            "skipped": True,
        }

    dataset_kwargs: dict[str, Any] = {
        "path": task.source,
        "split": task.split,
        "streaming": True,
        "revision": task.revision,
    }
    if task.subset not in (None, "default"):
        dataset_kwargs["name"] = task.subset
    dataset = load_dataset_with_backoff(
        context=f"{task.source} subset={subset_name} split={task.split} shard={task.shard_idx}",
        **dataset_kwargs,
    )
    shard_dataset = dataset.shard(num_shards=task.num_shards, index=task.shard_idx)

    counters: Counter[str] = Counter()
    result = write_jsonl_file(
        iter_code_instruction_documents(
            shard_dataset,
            task.cfg,
            subset=subset_name,
            split=task.split,
            counters=counters,
        ),
        output_filename,
    )
    logger.info(
        "Wrote %s rows to %s for subset=%s split=%s shard=%s counters=%s",
        result["count"],
        result["path"],
        subset_name,
        task.split,
        task.shard_idx,
        dict(counters),
    )

    return {
        "subset": subset_name,
        "split": task.split,
        "shard_idx": task.shard_idx,
        "path": result["path"],
        "count": result["count"],
        **dict(counters),
    }


@draccus.wrap()
def transform_code_instruction_midtraining(cfg: CodeInstructionMidtrainingConfig) -> str:
    """Transform configured code-instruction Hugging Face shards to Dolma documents."""

    all_tasks = list(get_code_instruction_shard_tasks(cfg))
    logger.info("Found %s code-instruction shards for %s", len(all_tasks), unwrap_versioned_value(cfg.source))

    metrics_path = os.path.join(cfg.output_path, "metrics")
    pipeline = (
        Dataset.from_list(all_tasks)
        .map(process_code_instruction_shard_task)
        .write_jsonl(f"{metrics_path}/{{shard:05d}}-transform.jsonl", skip_existing=True)
    )
    ctx_kwargs: dict[str, Any] = {"name": "transform-code-instruction-midtraining"}
    max_parallelism = unwrap_versioned_value(cfg.max_parallelism)
    if max_parallelism is not None:
        ctx_kwargs["max_workers"] = max_parallelism
    metric_files = ZephyrContext(**ctx_kwargs).execute(pipeline).results

    by_subset_split: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for metric_file in metric_files:
        result = next(iter(load_jsonl(metric_file)))
        by_subset_split[(result["subset"], result["split"])].append(result)

    for (subset, split), shard_results in sorted(by_subset_split.items()):
        total_count = sum(result["count"] for result in shard_results)
        total_input = sum(result.get("input_rows", 0) for result in shard_results)
        logger.info("Wrote %s/%s rows for %s/%s", total_count, total_input, subset, split)

    return cfg.output_path


if __name__ == "__main__":
    transform_code_instruction_midtraining()
