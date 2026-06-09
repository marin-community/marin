# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Long-context reading and retrieval PPL slices.

Issue #5825 (parent #5819) calls for long-context validation coverage that
moves before e19/e20 flops while staying cheap enough for periodic tracking,
and explicitly excludes AA-LCR documents/questions or other held-out benchmark
items from core tracking.

Two slice families are exposed here, both backed by public Hugging Face
mirrors that are not part of AA-LCR:

- Raw long-doc PPL over PG19, GovReport, and QuALITY contexts.
- Target-only retrieval PPL over converted QASPER and NarrativeQA mirrors.
- Target-only summarization PPL over BookSum validation/test chapters.

The slice set is intentionally limited. SEC EDGAR documents are deferred until
the TeraflopAI/SEC-EDGAR datakit source (#5305) lands so we share one
materialization.
"""

from __future__ import annotations

import json
import posixpath
from typing import Any

import fsspec
from datasets import load_dataset
from marin.evaluation.perplexity_gap import RawTextEvaluationDataset, raw_text_dataset, supervised_text_dataset
from marin.execution.step_spec import StepSpec
from marin.processing.tokenize import HfDatasetSpec
from marin.utils import fsspec_mkdirs

LONG_CONTEXT_EPIC = 5819
LONG_CONTEXT_ISSUE = 5825
FAMILY = "long_context"
QASPER_DATASET_ID = "urialon/converted_qasper"
QASPER_VALIDATION_SPLIT = "validation"
QASPER_STAGED_OUTPUT_FILENAME = "staged.jsonl.gz"
BOOKSUM_DATASET_ID = "kmfoda/booksum"
BOOKSUM_CONFIG_NAME = "default"
BOOKSUM_VALIDATION_SPLIT = "validation"
BOOKSUM_TEST_SPLIT = "test"
BOOKSUM_STAGED_OUTPUT_FILENAME = "staged.jsonl.gz"
BOOKSUM_SOURCE_SPLITS = frozenset((BOOKSUM_VALIDATION_SPLIT, BOOKSUM_TEST_SPLIT))
_QASPER_TEXT_MARKER = "\nText: "


def _registry_key(name: str) -> str:
    return posixpath.join(FAMILY, name)


def _tags(split: str, kind: str) -> tuple[str, ...]:
    return (
        FAMILY,
        f"epic:{LONG_CONTEXT_EPIC}",
        f"issue:{LONG_CONTEXT_ISSUE}",
        f"kind:{kind}",
        f"split:{split}",
    )


def render_qasper_answer_prompt(raw_input: str) -> str:
    """Render converted-QASPER rows with the question adjacent to the answer target."""

    if not raw_input.startswith("Q: "):
        raise ValueError("QASPER input must start with 'Q: '.")
    if _QASPER_TEXT_MARKER not in raw_input:
        raise ValueError("QASPER input must contain a '\\nText: ' marker.")

    question, text = raw_input[len("Q: ") :].split(_QASPER_TEXT_MARKER, maxsplit=1)
    question = question.strip()
    text = text.strip()
    if not question:
        raise ValueError("QASPER question is empty.")
    if not text:
        raise ValueError("QASPER context text is empty.")
    return f"Text:\n{text}\n\nQuestion:\n{question}\n\nAnswer:\n"


def qasper_supervised_record(row: dict[str, Any], *, row_index: int) -> dict[str, Any] | None:
    """Convert one converted-QASPER row into a supervised target-only record."""

    target = row.get("output")
    if not isinstance(target, str):
        raise ValueError(f"QASPER row {row_index} has non-string output.")
    target = target.strip()
    if not target:
        return None

    raw_input = row.get("input")
    if not isinstance(raw_input, str):
        raise ValueError(f"QASPER row {row_index} has non-string input.")

    return {
        "id": str(row.get("pid") or row.get("id") or row_index),
        "input": render_qasper_answer_prompt(raw_input),
        "target": target,
        "source": f"{QASPER_DATASET_ID}:{QASPER_VALIDATION_SPLIT}",
        "provenance": {
            "split": QASPER_VALIDATION_SPLIT,
            "row_index": row_index,
            "source_id": row.get("id"),
            "source_pid": row.get("pid"),
            "render": "text_question_answer_target_only",
        },
    }


def stage_qasper_supervised_source(output_path: str) -> dict[str, Any]:
    """Stage QASPER validation rows with an explicit answer prompt before the target."""

    fsspec_mkdirs(output_path, exist_ok=True)
    out_file = posixpath.join(output_path, QASPER_STAGED_OUTPUT_FILENAME)
    records = 0
    with fsspec.open(out_file, "wt", compression="gzip") as sink:
        rows = load_dataset(QASPER_DATASET_ID, split=QASPER_VALIDATION_SPLIT, streaming=True)
        for row_index, row in enumerate(rows):
            record = qasper_supervised_record(row, row_index=row_index)
            if record is None:
                continue
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
            records += 1

    fs, fs_path = fsspec.core.url_to_fs(out_file)
    return {"records": records, "output_file": out_file, "bytes_written": int(fs.info(fs_path)["size"])}


qasper_staged = StepSpec(
    name="evaluation/long_context_ppl/qasper_answer_prompt_v1",
    fn=stage_qasper_supervised_source,
    hash_attrs={
        "dataset_id": QASPER_DATASET_ID,
        "split": QASPER_VALIDATION_SPLIT,
        "render": "text_question_answer_target_only",
        "output_filename": QASPER_STAGED_OUTPUT_FILENAME,
    },
)


def render_booksum_summary_prompt(row: dict[str, Any], *, row_index: int) -> str:
    """Render BookSum rows with a clear summary target boundary."""

    chapter = row.get("chapter")
    if not isinstance(chapter, str):
        raise ValueError(f"BookSum row {row_index} has non-string chapter.")
    chapter = chapter.strip()
    if not chapter:
        raise ValueError(f"BookSum row {row_index} has empty chapter.")

    return f"Chapter:\n{chapter}\n\nSummary:\n"


def booksum_supervised_record(row: dict[str, Any], *, row_index: int, split: str) -> dict[str, Any] | None:
    """Convert one BookSum row into a supervised target-only summarization record."""

    if split not in BOOKSUM_SOURCE_SPLITS:
        raise ValueError(f"Unsupported BookSum split: {split}")

    target = row.get("summary_text")
    if not isinstance(target, str):
        raise ValueError(f"BookSum row {row_index} has non-string summary_text.")
    target = target.strip()
    if not target:
        return None

    return {
        "id": str(row.get("book_id") or row.get("summary_path") or row_index),
        "input": render_booksum_summary_prompt(row, row_index=row_index),
        "target": target,
        "source": f"{BOOKSUM_DATASET_ID}:{split}",
        "provenance": {
            "split": split,
            "row_index": row_index,
            "source": row.get("source"),
            "book_id": row.get("book_id"),
            "summary_id": row.get("summary_id"),
            "summary_path": row.get("summary_path"),
            "render": "chapter_summary_target_only",
        },
    }


def stage_booksum_supervised_source(output_path: str, *, split: str) -> dict[str, Any]:
    """Stage BookSum rows with an explicit summary prompt before the target."""

    if split not in BOOKSUM_SOURCE_SPLITS:
        raise ValueError(f"Unsupported BookSum split: {split}")

    fsspec_mkdirs(output_path, exist_ok=True)
    out_file = posixpath.join(output_path, BOOKSUM_STAGED_OUTPUT_FILENAME)
    records = 0
    with fsspec.open(out_file, "wt", compression="gzip") as sink:
        rows = load_dataset(BOOKSUM_DATASET_ID, BOOKSUM_CONFIG_NAME, split=split, streaming=True)
        for row_index, row in enumerate(rows):
            record = booksum_supervised_record(row, row_index=row_index, split=split)
            if record is None:
                continue
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
            records += 1

    fs, fs_path = fsspec.core.url_to_fs(out_file)
    return {"records": records, "output_file": out_file, "bytes_written": int(fs.info(fs_path)["size"])}


def stage_booksum_validation_source(output_path: str) -> dict[str, Any]:
    return stage_booksum_supervised_source(output_path, split=BOOKSUM_VALIDATION_SPLIT)


def stage_booksum_test_source(output_path: str) -> dict[str, Any]:
    return stage_booksum_supervised_source(output_path, split=BOOKSUM_TEST_SPLIT)


booksum_validation_staged = StepSpec(
    name="evaluation/long_context_ppl/booksum_validation_summary_prompt_v1",
    fn=stage_booksum_validation_source,
    hash_attrs={
        "dataset_id": BOOKSUM_DATASET_ID,
        "config_name": BOOKSUM_CONFIG_NAME,
        "split": BOOKSUM_VALIDATION_SPLIT,
        "render": "chapter_summary_target_only",
        "output_filename": BOOKSUM_STAGED_OUTPUT_FILENAME,
    },
)


booksum_test_staged = StepSpec(
    name="evaluation/long_context_ppl/booksum_test_summary_prompt_v1",
    fn=stage_booksum_test_source,
    hash_attrs={
        "dataset_id": BOOKSUM_DATASET_ID,
        "config_name": BOOKSUM_CONFIG_NAME,
        "split": BOOKSUM_TEST_SPLIT,
        "render": "chapter_summary_target_only",
        "output_filename": BOOKSUM_STAGED_OUTPUT_FILENAME,
    },
)


def long_context_raw_validation_sets() -> dict[str, RawTextEvaluationDataset]:
    """Raw long-document PPL slices."""
    return {
        _registry_key("pg19_test"): raw_text_dataset(
            HfDatasetSpec(id="emozilla/pg19-test"),
            text_key="text",
            split="test",
            tags=_tags("test", "raw_long_doc"),
        ),
        _registry_key("govreport_validation"): raw_text_dataset(
            HfDatasetSpec(id="ccdv/govreport-summarization"),
            text_key="report",
            split="validation",
            tags=_tags("validation", "raw_long_doc"),
        ),
        _registry_key("quality_context_test"): raw_text_dataset(
            HfDatasetSpec(id="rbiswasfc/scrolls-quality-mcq"),
            text_key="context",
            split="test",
            tags=_tags("test", "raw_long_doc"),
        ),
    }


def long_context_supervised_validation_sets() -> dict[str, RawTextEvaluationDataset]:
    """Target-only long-context task PPL slices."""
    return {
        _registry_key("scrolls_qasper"): supervised_text_dataset(
            qasper_staged.as_executor_step().cd(QASPER_STAGED_OUTPUT_FILENAME),
            tags=(*_tags("validation", "supervised_qa"), "format:text_question_answer_target_only"),
        ),
        _registry_key("scrolls_narrative_qa"): supervised_text_dataset(
            HfDatasetSpec(id="mattercalm/narrative_qa"),
            input_key="input",
            target_key="output",
            split="validation",
            tags=_tags("validation", "supervised_qa"),
        ),
        _registry_key("booksum_validation"): supervised_text_dataset(
            booksum_validation_staged.as_executor_step().cd(BOOKSUM_STAGED_OUTPUT_FILENAME),
            tags=(
                *_tags("validation", "supervised_summarization"),
                "format:chapter_summary_target_only",
            ),
        ),
        _registry_key("booksum_test"): supervised_text_dataset(
            booksum_test_staged.as_executor_step().cd(BOOKSUM_STAGED_OUTPUT_FILENAME),
            tags=(*_tags("test", "supervised_summarization"), "format:chapter_summary_target_only"),
        ),
    }


def long_context_validation_sets() -> dict[str, RawTextEvaluationDataset]:
    """Union of raw and supervised long-context slices."""
    return {**long_context_raw_validation_sets(), **long_context_supervised_validation_sets()}
