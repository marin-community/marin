# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""AceCode-V2 prompt/test source materialization for verified coding SFT.

The upstream ``TIGER-Lab/AceCode-V2-122K`` dataset contains coding prompts and
unit tests, but no assistant solutions. This module keeps that source object
intact for candidate generation and verification. A later derived
``question -> solution`` artifact can be registered as ordinary SFT once
solutions have been generated and verified.
"""

import hashlib
import json
import logging
import posixpath
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from enum import StrEnum
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from marin.datakit.download.huggingface import download_hf_step
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from rigging.filesystem import open_url, url_to_fs

logger = logging.getLogger(__name__)

ACECODE_V2_HF_ID = "TIGER-Lab/AceCode-V2-122K"
ACECODE_V2_REVISION = "1b0c2ac0f2cbbc8ada5397c075bd17ccee441e50"
ACECODE_V2_REVISION_SHORT = ACECODE_V2_REVISION[:7]
ACECODE_V2_TRAIN_PARQUET_GLOB = "data/train-*.parquet"

ACECODE_V2_SOURCE_LABELS = ("evol", "oss", "stack_python_fns")
ACECODE_V2_VALIDATION_BUCKETS = 5
ACECODE_V2_HASH_BUCKETS = 100
ACECODE_V2_SMALL_SMOKE_PER_SOURCE = 100

ACECODE_V2_SOURCE_COLUMNS = (
    "id",
    "source",
    "split",
    "question",
    "tests",
    "num_tests",
    "question_sha256",
    "test_sha256",
    "upstream_dataset",
    "upstream_revision",
)

ACECODE_V2_SOURCE_SCHEMA = pa.schema(
    [
        pa.field("id", pa.string(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("split", pa.string(), nullable=False),
        pa.field("question", pa.string(), nullable=False),
        pa.field("tests", pa.list_(pa.string()), nullable=False),
        pa.field("num_tests", pa.int64(), nullable=False),
        pa.field("question_sha256", pa.string(), nullable=False),
        pa.field("test_sha256", pa.string(), nullable=False),
        pa.field("upstream_dataset", pa.string(), nullable=False),
        pa.field("upstream_revision", pa.string(), nullable=False),
    ]
)


class AceCodeV2Split(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"


class AceCodeV2View(StrEnum):
    TRAIN_ALL_BUT_HOLDOUT = "train_all_but_holdout"
    VALIDATION_5PCT_SOURCE_STRATIFIED = "validation_5pct_source_stratified"
    SMALL_SMOKE_300 = "small_smoke_300"


DEFAULT_PROMPT_TEST_VIEWS = (
    AceCodeV2View.TRAIN_ALL_BUT_HOLDOUT,
    AceCodeV2View.SMALL_SMOKE_300,
)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_test_sha256(tests: Sequence[str]) -> str:
    payload = json.dumps(list(tests), sort_keys=True, separators=(",", ":"))
    return sha256_text(payload)


def acecode_v2_split(row_id: str) -> AceCodeV2Split:
    bucket = int(hashlib.sha256(row_id.encode("utf-8")).hexdigest()[:8], 16) % ACECODE_V2_HASH_BUCKETS
    if bucket < ACECODE_V2_VALIDATION_BUCKETS:
        return AceCodeV2Split.VALIDATION
    return AceCodeV2Split.TRAIN


def acecode_v2_source_row(row: Mapping[str, Any]) -> dict[str, Any]:
    row_id = _required_text(row, "id")
    source = _required_source(row)
    question = _required_text(row, "question", strip=False)
    tests = _required_tests(row)

    return {
        "id": row_id,
        "source": source,
        "split": acecode_v2_split(row_id).value,
        "question": question,
        "tests": tests,
        "num_tests": len(tests),
        "question_sha256": sha256_text(question),
        "test_sha256": stable_test_sha256(tests),
        "upstream_dataset": ACECODE_V2_HF_ID,
        "upstream_revision": ACECODE_V2_REVISION,
    }


def prompt_test_view_rows(
    rows: Sequence[Mapping[str, Any]],
    view: AceCodeV2View,
    *,
    small_smoke_per_source: int = ACECODE_V2_SMALL_SMOKE_PER_SOURCE,
) -> list[dict[str, Any]]:
    if view == AceCodeV2View.TRAIN_ALL_BUT_HOLDOUT:
        return _rows_for_split(rows, AceCodeV2Split.TRAIN)

    if view == AceCodeV2View.VALIDATION_5PCT_SOURCE_STRATIFIED:
        return _rows_for_split(rows, AceCodeV2Split.VALIDATION)

    if view == AceCodeV2View.SMALL_SMOKE_300:
        return source_balanced_rows(
            rows,
            per_source=small_smoke_per_source,
            split=AceCodeV2Split.TRAIN,
        )

    raise ValueError(f"Unknown AceCode-V2 view: {view}")


def source_balanced_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    per_source: int,
    split: AceCodeV2Split | None,
) -> list[dict[str, Any]]:
    if per_source < 1:
        raise ValueError(f"per_source must be positive, got {per_source}")

    selected: list[dict[str, Any]] = []
    for source in ACECODE_V2_SOURCE_LABELS:
        source_rows = []
        for row in rows:
            if row["source"] != source:
                continue
            if split is not None and row["split"] != split.value:
                continue
            source_rows.append(dict(row))
        source_rows.sort(key=_stable_source_row_key)
        if len(source_rows) < per_source:
            raise ValueError(
                f"Cannot build source-balanced AceCode-V2 view: source={source!r} has "
                f"{len(source_rows)} rows, need {per_source}."
            )
        selected.extend(source_rows[:per_source])

    selected.sort(key=lambda row: (row["source"], _stable_source_row_key(row)))
    return selected


def materialize_prompt_test_views(
    input_path: str,
    output_path: str,
    *,
    views: Sequence[AceCodeV2View] = DEFAULT_PROMPT_TEST_VIEWS,
    small_smoke_per_source: int = ACECODE_V2_SMALL_SMOKE_PER_SOURCE,
) -> None:
    rows = sorted(acecode_v2_source_rows(input_path), key=_stable_source_row_key)
    if not rows:
        raise ValueError(f"No AceCode-V2 rows found under {input_path}")

    manifest_views: dict[str, dict[str, Any]] = {}
    for view in views:
        view_rows = prompt_test_view_rows(rows, view, small_smoke_per_source=small_smoke_per_source)
        _write_view(output_path, view, view_rows)
        manifest_views[view.value] = _view_manifest(view_rows)
        logger.info("Wrote AceCode-V2 %s view with %d rows", view.value, len(view_rows))

    _write_manifest(output_path, manifest_views)


def acecode_v2_source_rows(input_path: str) -> Iterable[dict[str, Any]]:
    for parquet_url in _parquet_urls(input_path):
        with open_url(parquet_url, "rb") as handle:
            table = pq.read_table(handle)
        for row in table.to_pylist():
            yield acecode_v2_source_row(row)


def acecode_v2_raw_step() -> StepSpec:
    return download_hf_step(
        "raw/acecode-v2-122k",
        hf_dataset_id=ACECODE_V2_HF_ID,
        revision=ACECODE_V2_REVISION,
        hf_urls_glob=[ACECODE_V2_TRAIN_PARQUET_GLOB],
        override_output_path=f"raw/acecode-v2-122k-{ACECODE_V2_REVISION_SHORT}",
    )


def acecode_v2_prompt_test_views_step(raw_step: StepSpec | None = None) -> StepSpec:
    raw = raw_step if raw_step is not None else acecode_v2_raw_step()
    return StepSpec(
        name="posttrain/acecode-v2-122k/prompt-test-views",
        deps=[raw],
        hash_attrs={
            "version": "v1",
            "upstream_dataset": ACECODE_V2_HF_ID,
            "upstream_revision": ACECODE_V2_REVISION,
            "views": [view.value for view in DEFAULT_PROMPT_TEST_VIEWS],
            "validation_buckets": ACECODE_V2_VALIDATION_BUCKETS,
            "hash_buckets": ACECODE_V2_HASH_BUCKETS,
            "small_smoke_per_source": ACECODE_V2_SMALL_SMOKE_PER_SOURCE,
        },
        fn=lambda output_path: materialize_prompt_test_views(
            input_path=raw.output_path,
            output_path=output_path,
        ),
        override_output_path=f"posttrain/acecode-v2-122k-{ACECODE_V2_REVISION_SHORT}/prompt-test-views-v1",
    )


def build_steps() -> list[StepSpec]:
    raw = acecode_v2_raw_step()
    prompt_test_views = acecode_v2_prompt_test_views_step(raw)
    return [raw, prompt_test_views]


def main() -> None:
    StepRunner().run(build_steps())


def _required_text(row: Mapping[str, Any], key: str, *, strip: bool = True) -> str:
    value = row.get(key)
    if not isinstance(value, str):
        raise ValueError(f"AceCode-V2 row has non-string {key}: {type(value).__name__}")
    if not value.strip():
        raise ValueError(f"AceCode-V2 row has empty {key}")
    if strip:
        return value.strip()
    return value


def _required_source(row: Mapping[str, Any]) -> str:
    source = _required_text(row, "source")
    if source not in ACECODE_V2_SOURCE_LABELS:
        raise ValueError(f"Unexpected AceCode-V2 source {source!r}")
    return source


def _required_tests(row: Mapping[str, Any]) -> list[str]:
    value = row.get("tests")
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError(f"AceCode-V2 row has non-list tests: {type(value).__name__}")

    tests = list(value)
    if not tests:
        raise ValueError("AceCode-V2 row has empty tests")

    for test in tests:
        if not isinstance(test, str) or not test.strip():
            raise ValueError("AceCode-V2 row has a non-string or empty test")

    return tests


def _rows_for_split(rows: Sequence[Mapping[str, Any]], split: AceCodeV2Split) -> list[dict[str, Any]]:
    return [dict(row) for row in rows if row["split"] == split.value]


def _stable_source_row_key(row: Mapping[str, Any]) -> str:
    return sha256_text(str(row["id"]))


def _parquet_urls(input_path: str) -> list[str]:
    fs, resolved_path = url_to_fs(input_path)
    pattern = posixpath.join(resolved_path.rstrip("/"), "**", "*.parquet")
    matches = sorted(fs.glob(pattern))
    if not matches:
        raise ValueError(f"No parquet files found under {input_path}")
    return [fs.unstrip_protocol(match) for match in matches]


def _write_view(output_path: str, view: AceCodeV2View, rows: Sequence[Mapping[str, Any]]) -> None:
    view_dir = posixpath.join(output_path.rstrip("/"), view.value)
    fs, resolved_view_dir = url_to_fs(view_dir)
    fs.makedirs(resolved_view_dir, exist_ok=True)

    table = pa.Table.from_pylist([dict(row) for row in rows], schema=ACECODE_V2_SOURCE_SCHEMA)
    output_file = posixpath.join(view_dir, "data-00000-of-00001.parquet")
    with open_url(output_file, "wb") as handle:
        pq.write_table(table, handle)


def _view_manifest(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    source_counts = Counter(str(row["source"]) for row in rows)
    split_counts = Counter(str(row["split"]) for row in rows)
    return {
        "rows": len(rows),
        "sources": dict(sorted(source_counts.items())),
        "splits": dict(sorted(split_counts.items())),
    }


def _write_manifest(output_path: str, views: Mapping[str, Mapping[str, Any]]) -> None:
    fs, resolved_output_path = url_to_fs(output_path)
    fs.makedirs(resolved_output_path, exist_ok=True)

    manifest = {
        "upstream_dataset": ACECODE_V2_HF_ID,
        "upstream_revision": ACECODE_V2_REVISION,
        "source_schema": list(ACECODE_V2_SOURCE_COLUMNS),
        "views": dict(views),
    }
    manifest_path = posixpath.join(output_path.rstrip("/"), "manifest.json")
    with open_url(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
