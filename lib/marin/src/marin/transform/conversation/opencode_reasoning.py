# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Hydrate and transform NVIDIA OpenCodeReasoning-2 rows for SFT."""

import dataclasses
import hashlib
import json
import logging
import os
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

import draccus
from marin.core.conversation import DolmaConversationOutput, OpenAIChatMessage
from marin.transform.conversation.transform_conversation import DEFAULT_TEXT_REPLACEMENTS
from marin.utils import fsspec_glob, get_directory_friendly_name, load_dataset_with_backoff
from rigging.filesystem import url_to_fs
from zephyr import Dataset, ZephyrContext, load_jsonl, write_jsonl_file

logger = logging.getLogger(__name__)

OPENCODE_REASONING_2_DATASET_ID = "nvidia/OpenCodeReasoning-2"
OPENCODE_REASONING_2_REVISION = "eadf535931451525f3e5621d0f960c240bc62fd9"
OPENCODE_REASONING_2_CONFIG = "train"

OCR2_METADATA_COLUMNS = (
    "id",
    "question_id",
    "dataset",
    "split",
    "index",
    "source",
    "difficulty",
    "license",
    "judgement",
    "pass_rate",
)


class OCR2View(StrEnum):
    R1_GENERATION = "r1_generation"


class OCR2DropReason(StrEnum):
    EMPTY_ASSISTANT = "empty_assistant"
    EMPTY_PROMPT = "empty_prompt"
    FILTERED_JUDGEMENT = "filtered_judgement"
    FILTERED_PASS_RATE = "filtered_pass_rate"
    MISSING_HYDRATION = "missing_hydration"
    PLACEHOLDER_PROMPT = "placeholder_prompt"


@dataclass(frozen=True)
class OCR2SourceRevision:
    ocr2_dataset_value: str
    dataset: str
    revision: str


@dataclass(frozen=True)
class OCR2SourceSettings:
    source_license: str
    data_file_format: str | None = None
    data_files_template: str | None = None


OCR2_DEFAULT_SOURCE_REVISIONS = (
    OCR2SourceRevision("taco", "BAAI/TACO", "d593ed0a2becbbc952230bb89be09189bf1056dc"),
    OCR2SourceRevision("apps", "codeparrot/apps", "21e74ddf8de1a21436da12e3e653065c5213e9d1"),
    OCR2SourceRevision("code_contests", "deepmind/code_contests", "802411c3010cb00d1b05bad57ca77365a3c699d6"),
    OCR2SourceRevision("open-r1/codeforces", "open-r1/codeforces", "fbe3f6e903ee854eec2e69e9d96d0306cde59baf"),
)

OCR2_SOURCE_SETTINGS = {
    "taco": OCR2SourceSettings(
        source_license="apache-2.0",
        data_file_format="arrow",
        data_files_template="{split}/data-*.arrow",
    ),
    "apps": OCR2SourceSettings(
        source_license="mit",
        data_file_format="json",
        data_files_template="{split}.jsonl",
    ),
    "code_contests": OCR2SourceSettings(source_license="cc-by-4.0"),
    "open-r1/codeforces": OCR2SourceSettings(source_license="cc-by-4.0"),
}


@dataclass(frozen=True)
class OCR2HydrationConfig:
    output_path: str
    source_revisions: tuple[OCR2SourceRevision, ...]
    ocr2_revision: str = OPENCODE_REASONING_2_REVISION
    splits: tuple[str, ...] = ("python", "cpp")
    allow_missing_keys: tuple[str, ...] = ()
    max_parallelism: int | None = 32


@dataclass(frozen=True)
class OCR2TransformConfig:
    output_path: str
    hydration_path: str
    split: str
    view: OCR2View = OCR2View.R1_GENERATION
    ocr2_revision: str = OPENCODE_REASONING_2_REVISION
    require_hydration: bool = True
    allowed_judgements: tuple[str, ...] = ("right",)
    min_pass_rate: float | None = None
    max_parallelism: int | None = 32


@dataclass
class OCR2KeySummary:
    keys: set[str] = field(default_factory=set)
    rows_by_ocr2_split: Counter[str] = field(default_factory=Counter)
    duplicate_key_occurrences: int = 0
    question_ids_by_key: dict[str, str] = field(default_factory=dict)
    unexpected_datasets: Counter[str] = field(default_factory=Counter)


@dataclass(frozen=True)
class OCR2HydrationTask:
    ocr2_dataset_value: str
    source_dataset_id: str
    source_revision: str
    source_split: str
    source_license: str
    requested_indexes: tuple[str, ...]
    question_ids_by_index: dict[str, str]
    output_path: str


@dataclass(frozen=True)
class OCR2HydratedPrompt:
    key: str
    prompt: str
    source_dataset_id: str
    source_revision: str
    source_license: str


@dataclass(frozen=True)
class OCR2TransformResult:
    record: dict[str, Any] | None
    drop_reason: OCR2DropReason | None


@dataclass(frozen=True)
class OCR2TransformTask:
    split: str
    shard_idx: int
    num_shards: int
    cfg: OCR2TransformConfig


def _text_or_none(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
    else:
        text = str(value).strip()
    return text or None


def _apply_text_replacements(text: str, replacements: Mapping[str, str]) -> str:
    updated = text
    for old, new in replacements.items():
        updated = updated.replace(old, new)
    return updated


def ocr2_source_key(dataset: object, split: object, index: object) -> str:
    dataset_text = _text_or_none(dataset)
    split_text = _text_or_none(split)
    index_text = _text_or_none(index)
    if dataset_text is None or split_text is None or index_text is None:
        raise ValueError(f"OCR2 source key requires dataset, split, and index, got {dataset!r}, {split!r}, {index!r}")
    return f"{dataset_text}/{split_text}/{index_text}"


def parse_ocr2_source_key(key: str) -> tuple[str, str, str]:
    parts = key.rsplit("/", 2)
    if len(parts) != 3:
        raise ValueError(f"Invalid OCR2 source key: {key!r}")
    return parts[0], parts[1], parts[2]


def _update_ocr2_key_summary(summary: OCR2KeySummary, rows: Iterable[Mapping[str, object]], ocr2_split: str) -> None:
    for row in rows:
        summary.rows_by_ocr2_split[ocr2_split] += 1
        dataset_value = _text_or_none(row.get("dataset"))
        if dataset_value is not None and dataset_value not in OCR2_SOURCE_SETTINGS:
            summary.unexpected_datasets[dataset_value] += 1

        key = ocr2_source_key(row.get("dataset"), row.get("split"), row.get("index"))
        if key in summary.keys:
            summary.duplicate_key_occurrences += 1
        else:
            summary.keys.add(key)

        question_id = _text_or_none(row.get("question_id"))
        if question_id is not None and key not in summary.question_ids_by_key:
            summary.question_ids_by_key[key] = question_id


def collect_ocr2_question_keys_from_rows(
    rows: Iterable[Mapping[str, object]],
    ocr2_split: str = "unknown",
) -> OCR2KeySummary:
    summary = OCR2KeySummary()
    _update_ocr2_key_summary(summary, rows, ocr2_split)
    return summary


def _ocr2_split_rows(split: str, revision: str):
    return load_dataset_with_backoff(
        context=f"{OPENCODE_REASONING_2_DATASET_ID} split={split}",
        path=OPENCODE_REASONING_2_DATASET_ID,
        name=OPENCODE_REASONING_2_CONFIG,
        split=split,
        streaming=True,
        revision=revision,
    )


def _collect_ocr2_question_keys(config: OCR2HydrationConfig) -> OCR2KeySummary:
    summary = OCR2KeySummary()
    for split in config.splits:
        _update_ocr2_key_summary(summary, _ocr2_split_rows(split, config.ocr2_revision), split)
    return summary


def _source_revision_by_dataset(
    source_revisions: tuple[OCR2SourceRevision, ...],
) -> dict[str, OCR2SourceRevision]:
    if not source_revisions:
        raise ValueError("OCR2 hydration requires explicit source revisions.")

    revisions: dict[str, OCR2SourceRevision] = {}
    for source_revision in source_revisions:
        if source_revision.ocr2_dataset_value in revisions:
            raise ValueError(f"Duplicate OCR2 source revision for {source_revision.ocr2_dataset_value!r}")
        if source_revision.ocr2_dataset_value not in OCR2_SOURCE_SETTINGS:
            raise ValueError(f"Unsupported OCR2 source dataset: {source_revision.ocr2_dataset_value!r}")
        revisions[source_revision.ocr2_dataset_value] = source_revision
    return revisions


def _hydration_tasks(config: OCR2HydrationConfig, summary: OCR2KeySummary) -> list[OCR2HydrationTask]:
    source_revisions = _source_revision_by_dataset(config.source_revisions)
    keys_by_source_split: dict[tuple[str, str], set[str]] = defaultdict(set)
    question_ids_by_source_split: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)

    for key in summary.keys:
        dataset_value, source_split, index = parse_ocr2_source_key(key)
        keys_by_source_split[(dataset_value, source_split)].add(index)
        question_id = summary.question_ids_by_key.get(key)
        if question_id is not None:
            question_ids_by_source_split[(dataset_value, source_split)][index] = question_id

    tasks: list[OCR2HydrationTask] = []
    for (dataset_value, source_split), indexes in sorted(keys_by_source_split.items()):
        source_revision = source_revisions.get(dataset_value)
        if source_revision is None:
            raise ValueError(f"OCR2 source dataset {dataset_value!r} has no pinned revision.")
        settings = OCR2_SOURCE_SETTINGS[dataset_value]
        tasks.append(
            OCR2HydrationTask(
                ocr2_dataset_value=dataset_value,
                source_dataset_id=source_revision.dataset,
                source_revision=source_revision.revision,
                source_split=source_split,
                source_license=settings.source_license,
                requested_indexes=tuple(
                    sorted(indexes, key=lambda value: (0, int(value)) if value.isdigit() else (1, value))
                ),
                question_ids_by_index=question_ids_by_source_split[(dataset_value, source_split)],
                output_path=config.output_path,
            )
        )
    return tasks


def extract_taco_prompt(row: Mapping[str, object]) -> str | None:
    return _text_or_none(row.get("question"))


def extract_apps_prompt(row: Mapping[str, object]) -> str | None:
    return _text_or_none(row.get("question"))


def extract_code_contests_prompt(row: Mapping[str, object]) -> str | None:
    return _text_or_none(row.get("description"))


def _append_codeforces_section(parts: list[str], title: str, content: object) -> None:
    text = _text_or_none(content)
    if text is not None:
        parts.append(f"{title}\n{text}")


def _coerce_example_values(value: object) -> list[object]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _format_codeforces_examples(examples: object) -> str | None:
    formatted: list[str] = []
    if isinstance(examples, Mapping):
        inputs = _coerce_example_values(examples.get("input") or examples.get("inputs"))
        outputs = _coerce_example_values(examples.get("output") or examples.get("outputs"))
        for idx, (example_input, example_output) in enumerate(zip(inputs, outputs, strict=False), start=1):
            input_text = _text_or_none(example_input)
            output_text = _text_or_none(example_output)
            if input_text is not None:
                formatted.append(f"Example {idx} Input\n{input_text}")
            if output_text is not None:
                formatted.append(f"Example {idx} Output\n{output_text}")
    elif isinstance(examples, (list, tuple)):
        for idx, example in enumerate(examples, start=1):
            if not isinstance(example, Mapping):
                continue
            input_text = _text_or_none(example.get("input"))
            output_text = _text_or_none(example.get("output"))
            if input_text is not None:
                formatted.append(f"Example {idx} Input\n{input_text}")
            if output_text is not None:
                formatted.append(f"Example {idx} Output\n{output_text}")

    if not formatted:
        return None
    return "\n\n".join(formatted)


def extract_codeforces_prompt(row: Mapping[str, object]) -> str | None:
    description = _text_or_none(row.get("description"))
    if description is None:
        return None

    parts = [description]
    _append_codeforces_section(parts, "Input", row.get("input_format"))
    _append_codeforces_section(parts, "Output", row.get("output_format"))
    examples = _format_codeforces_examples(row.get("examples"))
    if examples is not None:
        parts.append(f"Examples\n{examples}")
    _append_codeforces_section(parts, "Note", row.get("note"))
    return "\n\n".join(parts)


def extract_ocr2_prompt(dataset_value: str, row: Mapping[str, object]) -> str | None:
    if dataset_value == "taco":
        return extract_taco_prompt(row)
    if dataset_value == "apps":
        return extract_apps_prompt(row)
    if dataset_value == "code_contests":
        return extract_code_contests_prompt(row)
    if dataset_value == "open-r1/codeforces":
        return extract_codeforces_prompt(row)
    raise ValueError(f"Unsupported OCR2 source dataset: {dataset_value!r}")


def ocr2_source_load_kwargs(task: OCR2HydrationTask) -> dict[str, object]:
    settings = OCR2_SOURCE_SETTINGS[task.ocr2_dataset_value]
    if settings.data_file_format is not None:
        if settings.data_files_template is None:
            raise ValueError(f"OCR2 source {task.ocr2_dataset_value!r} has no data file template.")
        data_files = (
            f"hf://datasets/{task.source_dataset_id}@{task.source_revision}/"
            f"{settings.data_files_template.format(split=task.source_split)}"
        )
        kwargs: dict[str, object] = {
            "path": settings.data_file_format,
            "data_files": data_files,
            "split": "train",
            "streaming": True,
        }
    else:
        kwargs = {
            "path": task.source_dataset_id,
            "split": task.source_split,
            "streaming": True,
            "revision": task.source_revision,
        }
    return kwargs


def _source_rows(task: OCR2HydrationTask):
    return load_dataset_with_backoff(
        context=f"{task.source_dataset_id} split={task.source_split}",
        **ocr2_source_load_kwargs(task),
    )


def _hydration_output_filename(task: OCR2HydrationTask) -> str:
    dataset_name = get_directory_friendly_name(task.ocr2_dataset_value)
    split_name = get_directory_friendly_name(task.source_split)
    return os.path.join(task.output_path, "data", dataset_name, f"{split_name}.jsonl.gz")


def process_ocr2_hydration_task(task: OCR2HydrationTask) -> dict[str, object]:
    output_filename = _hydration_output_filename(task)
    requested_indexes = set(task.requested_indexes)
    found_indexes: set[str] = set()
    empty_prompt_keys: set[str] = set()
    source_rows_seen = 0

    def hydration_records() -> Iterator[dict[str, object]]:
        nonlocal source_rows_seen
        for row_idx, row in enumerate(_source_rows(task)):
            source_rows_seen += 1
            index = str(row_idx)
            if index not in requested_indexes:
                continue

            found_indexes.add(index)
            key = ocr2_source_key(task.ocr2_dataset_value, task.source_split, index)
            prompt = extract_ocr2_prompt(task.ocr2_dataset_value, row)
            if prompt is None:
                empty_prompt_keys.add(key)
            else:
                yield {
                    "key": key,
                    "dataset": task.ocr2_dataset_value,
                    "split": task.source_split,
                    "index": index,
                    "question_id": task.question_ids_by_index.get(index, ""),
                    "prompt": prompt,
                    "source_dataset_id": task.source_dataset_id,
                    "source_revision": task.source_revision,
                    "source_license": task.source_license,
                }

            if found_indexes == requested_indexes:
                break

    write_result = write_jsonl_file(hydration_records(), output_filename)
    missing_indexes = requested_indexes - found_indexes
    missing_keys = {ocr2_source_key(task.ocr2_dataset_value, task.source_split, index) for index in missing_indexes}
    return {
        "dataset": task.ocr2_dataset_value,
        "split": task.source_split,
        "source_dataset_id": task.source_dataset_id,
        "source_revision": task.source_revision,
        "requested_keys": len(requested_indexes),
        "hydrated_keys": write_result["count"],
        "found_source_rows": len(found_indexes),
        "source_rows_seen": source_rows_seen,
        "missing_keys": len(missing_keys),
        "missing_key_values": sorted(missing_keys),
        "empty_prompt_keys": len(empty_prompt_keys),
        "empty_prompt_key_values": sorted(empty_prompt_keys),
        "path": write_result["path"],
    }


def _load_jsonl_records(paths: Iterable[str]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in paths:
        records.extend(load_jsonl(path))
    return records


def _check_hydration_failures(
    metrics: Iterable[Mapping[str, object]],
    allow_missing_keys: tuple[str, ...],
) -> None:
    allowed = set(allow_missing_keys)
    bad_keys: set[str] = set()
    for metric in metrics:
        missing_keys = metric.get("missing_key_values", [])
        empty_prompt_keys = metric.get("empty_prompt_key_values", [])
        if isinstance(missing_keys, list):
            bad_keys.update(str(key) for key in missing_keys)
        if isinstance(empty_prompt_keys, list):
            bad_keys.update(str(key) for key in empty_prompt_keys)

    unallowed = sorted(bad_keys - allowed)
    if unallowed:
        sample = ", ".join(unallowed[:20])
        raise ValueError(f"OCR2 hydration missed {len(unallowed)} non-allowlisted keys. First misses: {sample}")


def _write_hydration_summary(
    output_path: str,
    ocr2_revision: str,
    key_summary: OCR2KeySummary,
    task_metrics: list[dict[str, object]],
    source_revisions: tuple[OCR2SourceRevision, ...],
) -> None:
    summary_path = os.path.join(output_path, "metrics", "summary.jsonl")
    summary_record = {
        "ocr2_revision": ocr2_revision,
        "rows_by_ocr2_split": dict(key_summary.rows_by_ocr2_split),
        "unique_requested_keys": len(key_summary.keys),
        "duplicate_key_occurrences": key_summary.duplicate_key_occurrences,
        "unexpected_datasets": dict(key_summary.unexpected_datasets),
        "source_revisions": [dataclasses.asdict(source_revision) for source_revision in source_revisions],
        "requested_keys": sum(int(metric.get("requested_keys", 0)) for metric in task_metrics),
        "hydrated_keys": sum(int(metric.get("hydrated_keys", 0)) for metric in task_metrics),
        "missing_keys": sum(int(metric.get("missing_keys", 0)) for metric in task_metrics),
        "empty_prompt_keys": sum(int(metric.get("empty_prompt_keys", 0)) for metric in task_metrics),
    }
    write_jsonl_file([summary_record], summary_path)


@draccus.wrap()
def hydrate_ocr2_questions(config: OCR2HydrationConfig) -> str:
    source_revisions = _source_revision_by_dataset(config.source_revisions)
    key_summary = _collect_ocr2_question_keys(config)
    if key_summary.unexpected_datasets:
        raise ValueError(f"Unsupported OCR2 source datasets in stream: {dict(key_summary.unexpected_datasets)}")

    missing_revision_labels = sorted(
        {parse_ocr2_source_key(key)[0] for key in key_summary.keys} - set(source_revisions.keys())
    )
    if missing_revision_labels:
        raise ValueError(f"OCR2 source revisions missing for: {missing_revision_labels}")

    tasks = _hydration_tasks(config, key_summary)
    logger.info("Hydrating %d OCR2 source split groups into %s", len(tasks), config.output_path)
    metrics_path = os.path.join(config.output_path, "metrics", "hydration-{shard:05d}.jsonl")
    pipeline = Dataset.from_list(tasks).map(process_ocr2_hydration_task).write_jsonl(metrics_path, skip_existing=True)
    ctx_kwargs: dict[str, object] = {"name": "ocr2-hydration"}
    if config.max_parallelism is not None:
        ctx_kwargs["max_workers"] = config.max_parallelism
    metric_files = ZephyrContext(**ctx_kwargs).execute(pipeline).results
    task_metrics = _load_jsonl_records(metric_files)

    _write_hydration_summary(
        config.output_path,
        config.ocr2_revision,
        key_summary,
        task_metrics,
        config.source_revisions,
    )
    _check_hydration_failures(task_metrics, config.allow_missing_keys)
    return config.output_path


def load_ocr2_hydration_cache(hydration_path: str) -> dict[str, OCR2HydratedPrompt]:
    pattern = os.path.join(hydration_path, "data", "**", "*.jsonl.gz")
    files = fsspec_glob(pattern)
    if not files:
        raise ValueError(f"No OCR2 hydration JSONL files found under {hydration_path!r}")

    prompts: dict[str, OCR2HydratedPrompt] = {}
    for file in files:
        for record in load_jsonl(file):
            key = str(record["key"])
            prompt = OCR2HydratedPrompt(
                key=key,
                prompt=str(record["prompt"]),
                source_dataset_id=str(record["source_dataset_id"]),
                source_revision=str(record["source_revision"]),
                source_license=str(record["source_license"]),
            )
            existing = prompts.get(key)
            if existing is not None and existing != prompt:
                raise ValueError(f"Conflicting OCR2 hydration records for key {key!r}")
            prompts[key] = prompt
    return prompts


def stable_ocr2_output_id(
    row: Mapping[str, object],
    prompt: str,
    assistant: str,
    view: OCR2View,
    ocr2_split: str,
) -> str:
    payload = {
        "row_id": row.get("id"),
        "ocr2_split": ocr2_split,
        "source_dataset": row.get("dataset"),
        "source_split": row.get("split"),
        "source_index": row.get("index"),
        "view": view.value,
        "prompt": prompt,
        "assistant": assistant,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _pass_rate_meets_threshold(raw_pass_rate: object, threshold: float) -> bool:
    text = _text_or_none(raw_pass_rate)
    if text is None:
        return False
    try:
        return float(text) >= threshold
    except ValueError:
        return False


def _metadata_from_row(row: Mapping[str, object], hydrated_prompt: OCR2HydratedPrompt) -> dict[str, object]:
    metadata = {column: row.get(column, "") for column in OCR2_METADATA_COLUMNS}
    metadata.update(
        {
            "hydrated_prompt_key": hydrated_prompt.key,
            "hydration_source_dataset_id": hydrated_prompt.source_dataset_id,
            "hydration_source_revision": hydrated_prompt.source_revision,
            "hydration_source_license": hydrated_prompt.source_license,
        }
    )
    return metadata


def transform_ocr2_row(
    row: Mapping[str, object],
    hydration_cache: Mapping[str, OCR2HydratedPrompt],
    config: OCR2TransformConfig,
) -> OCR2TransformResult:
    judgement = _text_or_none(row.get("judgement")) or ""
    if config.allowed_judgements and judgement not in set(config.allowed_judgements):
        return OCR2TransformResult(record=None, drop_reason=OCR2DropReason.FILTERED_JUDGEMENT)

    if config.min_pass_rate is not None and not _pass_rate_meets_threshold(row.get("pass_rate"), config.min_pass_rate):
        return OCR2TransformResult(record=None, drop_reason=OCR2DropReason.FILTERED_PASS_RATE)

    assistant = _text_or_none(row.get(config.view.value))
    if assistant is None:
        return OCR2TransformResult(record=None, drop_reason=OCR2DropReason.EMPTY_ASSISTANT)
    assistant = _apply_text_replacements(assistant, DEFAULT_TEXT_REPLACEMENTS)

    key = ocr2_source_key(row.get("dataset"), row.get("split"), row.get("index"))
    hydrated_prompt = hydration_cache.get(key)
    if hydrated_prompt is None:
        if config.require_hydration:
            raise ValueError(f"OCR2 row {row.get('id', '')!r} missing hydrated prompt for {key!r}")
        return OCR2TransformResult(record=None, drop_reason=OCR2DropReason.MISSING_HYDRATION)

    prompt = _text_or_none(hydrated_prompt.prompt)
    if prompt is None:
        return OCR2TransformResult(record=None, drop_reason=OCR2DropReason.EMPTY_PROMPT)
    if prompt == "-":
        return OCR2TransformResult(record=None, drop_reason=OCR2DropReason.PLACEHOLDER_PROMPT)

    output = DolmaConversationOutput(
        id=stable_ocr2_output_id(row, prompt, assistant, config.view, config.split),
        source=OPENCODE_REASONING_2_DATASET_ID,
        messages=[
            OpenAIChatMessage(role="user", content=prompt),
            OpenAIChatMessage(role="assistant", content=assistant),
        ],
        added=datetime.now(UTC).isoformat(),
        created="",
        metadata=_metadata_from_row(row, hydrated_prompt),
    )
    return OCR2TransformResult(record=output.model_dump(), drop_reason=None)


def _transform_output_filename(task: OCR2TransformTask) -> str:
    return os.path.join(task.cfg.output_path, f"shard_{task.shard_idx:05d}.jsonl.gz")


def _ocr2_transform_tasks(config: OCR2TransformConfig) -> list[OCR2TransformTask]:
    dataset = _ocr2_split_rows(config.split, config.ocr2_revision)
    num_shards = dataset.num_shards
    if not num_shards:
        raise ValueError(f"OCR2 split {config.split!r} does not expose streaming shards.")
    return [
        OCR2TransformTask(split=config.split, shard_idx=shard_idx, num_shards=num_shards, cfg=config)
        for shard_idx in range(num_shards)
    ]


def process_ocr2_transform_task(task: OCR2TransformTask) -> dict[str, object]:
    output_filename = _transform_output_filename(task)
    fs, _ = url_to_fs(output_filename)
    if fs.exists(output_filename):
        return {
            "split": task.split,
            "shard_idx": task.shard_idx,
            "path": output_filename,
            "input_rows": 0,
            "output_rows": 0,
            "skipped": True,
        }

    hydration_cache = load_ocr2_hydration_cache(task.cfg.hydration_path)
    dataset = _ocr2_split_rows(task.split, task.cfg.ocr2_revision)
    shard_dataset = dataset.shard(num_shards=task.num_shards, index=task.shard_idx)
    metrics: Counter[str] = Counter()

    def transformed_records() -> Iterator[dict[str, Any]]:
        for row in shard_dataset:
            metrics["input_rows"] += 1
            result = transform_ocr2_row(row, hydration_cache, task.cfg)
            if result.record is not None:
                metrics["output_rows"] += 1
                yield result.record
            elif result.drop_reason is not None:
                metrics[result.drop_reason.value] += 1

    write_result = write_jsonl_file(transformed_records(), output_filename)
    metrics["output_rows"] = int(write_result["count"])
    return {
        "split": task.split,
        "shard_idx": task.shard_idx,
        "path": write_result["path"],
        **dict(metrics),
    }


@draccus.wrap()
def transform_ocr2_sft(config: OCR2TransformConfig) -> str:
    tasks = _ocr2_transform_tasks(config)
    logger.info("Transforming %d OCR2 %s shards into %s", len(tasks), config.split, config.output_path)
    metrics_path = os.path.join(config.output_path, "metrics", "{shard:05d}-transform.jsonl")
    pipeline = Dataset.from_list(tasks).map(process_ocr2_transform_task).write_jsonl(metrics_path, skip_existing=True)
    ctx_kwargs: dict[str, object] = {"name": "ocr2-transform"}
    if config.max_parallelism is not None:
        ctx_kwargs["max_workers"] = config.max_parallelism
    metric_files = ZephyrContext(**ctx_kwargs).execute(pipeline).results

    task_metrics = _load_jsonl_records(metric_files)
    input_rows = sum(int(metric.get("input_rows", 0)) for metric in task_metrics)
    output_rows = sum(int(metric.get("output_rows", 0)) for metric in task_metrics)
    logger.info("OCR2 %s transform wrote %d rows from %d input rows", config.split, output_rows, input_rows)
    return config.output_path
