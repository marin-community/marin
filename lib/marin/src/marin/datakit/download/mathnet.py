# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""ShadenA/MathNet text-only SFT dataset download and transform.

The first MathNet view keeps only examples that can be rendered as text:
problem markdown in the user turn and one official solution in the assistant
turn. Rows with attached images are left for a later multimodal pipeline.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

import fsspec
from fray import ResourceConfig
from zephyr import Dataset, ZephyrContext, counters, load_parquet

from marin.core.conversation import DolmaConversationOutput, OpenAIChatMessage
from marin.datakit.download.huggingface import download_hf_step
from marin.execution.step_spec import StepSpec

HF_DATASET_ID = "ShadenA/MathNet"
HF_REVISION = "ae12e35eef0fc52bbbef270d6ef0f5b002252eb9"
HF_URLS_GLOB = ["data/all/train-*.parquet"]
LICENSE = "cc-by-4.0"
ATTACHED_IMAGE_MARKER = "attached_image_"
TEXT_SFT_VIEW = "text_sft_primary"
RAW_STEP_NAME = "raw/mathnet-v0"
PROCESSED_STEP_NAME = "processed/mathnet-v0/text-sft-primary"


class MathNetLanguagePolicy(StrEnum):
    ENGLISH_OR_UNKNOWN = "english_or_unknown"
    ALL_LANGUAGES = "all_languages"


class MathNetSolutionPolicy(StrEnum):
    FIRST = "first"
    ALL = "all"


@dataclass(frozen=True)
class MathNetTextSftConfig:
    raw_input_path: str
    output_path: str
    language_policy: MathNetLanguagePolicy = MathNetLanguagePolicy.ENGLISH_OR_UNKNOWN
    solution_policy: MathNetSolutionPolicy = MathNetSolutionPolicy.FIRST
    excluded_ids_path: str = ""


def has_attached_image_reference(text: str | None) -> bool:
    return bool(text and ATTACHED_IMAGE_MARKER in text)


def row_requires_image(row: Mapping[str, Any]) -> bool:
    if row.get("images"):
        return True
    return has_attached_image_reference(row.get("problem_markdown"))


def keep_language(language: str | None, policy: MathNetLanguagePolicy) -> bool:
    if policy == MathNetLanguagePolicy.ALL_LANGUAGES:
        return True
    return language in (None, "", "en", "English")


def selected_solutions(row: Mapping[str, Any], policy: MathNetSolutionPolicy) -> list[tuple[int, str]]:
    solutions = [solution.strip() for solution in row.get("solutions_markdown") or []]
    indexed = [(idx, solution) for idx, solution in enumerate(solutions) if solution]
    if policy == MathNetSolutionPolicy.FIRST:
        return indexed[:1]
    return indexed


def _load_excluded_ids(excluded_ids_path: str) -> set[str]:
    if not excluded_ids_path:
        return set()

    with fsspec.open(excluded_ids_path, "rt", encoding="utf-8") as handle:
        return {line.strip() for line in handle if line.strip()}


def _conversation_id(mathnet_id: str, solution_index: int) -> str:
    return f"mathnet:{HF_REVISION}:{mathnet_id}:solution-{solution_index}"


def row_to_text_sft_records(
    row: Mapping[str, Any],
    cfg: MathNetTextSftConfig,
    excluded_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    mathnet_id = str(row.get("id") or "").strip()
    if not mathnet_id:
        counters.increment("mathnet/skipped_empty_id")
        return []

    if mathnet_id in (excluded_ids or set()):
        counters.increment("mathnet/skipped_benchmark_exclusion")
        return []

    problem = (row.get("problem_markdown") or "").strip()
    if not problem:
        counters.increment("mathnet/skipped_empty_problem")
        return []

    if row_requires_image(row):
        counters.increment("mathnet/skipped_image")
        return []

    language = row.get("language")
    if not keep_language(language, cfg.language_policy):
        counters.increment("mathnet/skipped_language")
        return []

    solutions = selected_solutions(row, cfg.solution_policy)
    if not solutions:
        counters.increment("mathnet/skipped_empty_solution")
        return []

    records: list[dict[str, Any]] = []
    solution_count = len([solution for solution in row.get("solutions_markdown") or [] if solution and solution.strip()])
    for solution_index, solution in solutions:
        if has_attached_image_reference(solution):
            counters.increment("mathnet/skipped_image")
            continue

        output = DolmaConversationOutput(
            id=_conversation_id(mathnet_id, solution_index),
            source=HF_DATASET_ID,
            messages=[
                OpenAIChatMessage(role="user", content=problem),
                OpenAIChatMessage(role="assistant", content=solution),
            ],
            added=datetime.now(UTC).isoformat(),
            created="",
            metadata={
                "hf_revision": HF_REVISION,
                "license": LICENSE,
                "mathnet_id": mathnet_id,
                "country": row.get("country"),
                "competition": row.get("competition"),
                "topics_flat": row.get("topics_flat") or [],
                "language": language,
                "problem_type": row.get("problem_type"),
                "final_answer": row.get("final_answer"),
                "solution_index": solution_index,
                "solution_count": solution_count,
                "view": TEXT_SFT_VIEW,
                "has_images": False,
                "language_policy": cfg.language_policy.value,
                "solution_policy": cfg.solution_policy.value,
                "excluded_ids_path": cfg.excluded_ids_path,
            },
        )
        records.append(output.model_dump(mode="json"))

    if records:
        counters.increment("mathnet/kept", len(records))
    return records


def transform_text_sft(cfg: MathNetTextSftConfig) -> None:
    excluded_ids = _load_excluded_ids(cfg.excluded_ids_path)
    pipeline = (
        Dataset.from_files(f"{cfg.raw_input_path}/data/all/*.parquet")
        .flat_map(load_parquet)
        .flat_map(lambda row: row_to_text_sft_records(row, cfg, excluded_ids))
        .write_jsonl(
            f"{cfg.output_path}/data-{{shard:05d}}-of-{{total:05d}}.jsonl.gz",
            skip_existing=True,
        )
    )
    ctx = ZephyrContext(name="mathnet-text-sft-transform", resources=ResourceConfig(cpu=1, ram="8g"))
    ctx.execute(pipeline)


def download_mathnet_raw_step() -> StepSpec:
    """Download the deduplicated MathNet v0 ``all`` parquet shards."""
    return download_hf_step(
        RAW_STEP_NAME,
        hf_dataset_id=HF_DATASET_ID,
        revision=HF_REVISION,
        hf_urls_glob=HF_URLS_GLOB,
        worker_resources=ResourceConfig(cpu=1, ram="8g", disk="5g"),
    )


def mathnet_text_sft_primary_step(
    *,
    language_policy: MathNetLanguagePolicy = MathNetLanguagePolicy.ENGLISH_OR_UNKNOWN,
    solution_policy: MathNetSolutionPolicy = MathNetSolutionPolicy.FIRST,
    excluded_ids_path: str = "",
) -> StepSpec:
    """Create the text-only MathNet SFT processed-data step."""
    raw = download_mathnet_raw_step()
    return StepSpec(
        name=PROCESSED_STEP_NAME,
        deps=[raw],
        fn=lambda output_path: transform_text_sft(
            MathNetTextSftConfig(
                raw_input_path=raw.output_path,
                output_path=output_path,
                language_policy=language_policy,
                solution_policy=solution_policy,
                excluded_ids_path=excluded_ids_path,
            )
        ),
        hash_attrs={
            "hf_dataset_id": HF_DATASET_ID,
            "hf_revision": HF_REVISION,
            "hf_urls_glob": HF_URLS_GLOB,
            "language_policy": language_policy.value,
            "solution_policy": solution_policy.value,
            "excluded_ids_path": excluded_ids_path,
            "version": "v1",
        },
    )
