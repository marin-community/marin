# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""AI-MO/NuminaMath-1.5 dataset download and transform.

NuminaMath-1.5 is a curated math post-training corpus with CoT-formatted
solutions and explicit problem/solution validity metadata. This transform keeps
only rows marked valid by both fields, renders problem/solution pairs as tagged
transcripts, and preserves source metadata for downstream mixture analysis.
"""

import hashlib

from fray.types import ResourceConfig
from zephyr import counters
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext
from zephyr.readers import load_parquet

from marin.datakit.download.huggingface import download_hf_step
from marin.datakit.normalize import normalize_step
from marin.execution.step_spec import StepSpec

HF_DATASET_ID = "AI-MO/NuminaMath-1.5"
HF_REVISION = "1b05109"
TRAIN_PARQUET_GLOB = "data/train-*.parquet"
VALID_STATUS = "Yes"


def _clean_text(row: dict, key: str) -> str | None:
    value = row.get(key)
    if not isinstance(value, str):
        return None

    text = value.strip()
    if not text:
        return None

    return text


def _optional_text(row: dict, key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str):
        return ""

    return value.strip()


def _is_valid_row(row: dict) -> bool:
    return row.get("problem_is_valid") == VALID_STATUS and row.get("solution_is_valid") == VALID_STATUS


def row_to_doc(row: dict) -> list[dict]:
    problem = _clean_text(row, "problem")
    solution = _clean_text(row, "solution")
    if problem is None or solution is None:
        counters.pipeline.update_counter("numinamath_v1_5/dropped_empty", 1)
        return []

    if not _is_valid_row(row):
        counters.pipeline.update_counter("numinamath_v1_5/dropped_invalid", 1)
        return []

    text = f"<user>\n{problem}\n</user>\n\n<assistant>\n{solution}\n</assistant>"

    counters.pipeline.update_counter("numinamath_v1_5/kept", 1)
    return [
        {
            "id": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "problem_hash": hashlib.sha256(problem.encode("utf-8")).hexdigest(),
            "text": text,
            "source": HF_DATASET_ID,
            "numina_source": _optional_text(row, "source"),
            "answer": _optional_text(row, "answer"),
            "problem_type": _optional_text(row, "problem_type"),
            "question_type": _optional_text(row, "question_type"),
            "synthetic": row.get("synthetic") is True,
        }
    ]


def transform(input_path: str, output_path: str) -> None:
    pipeline = (
        Dataset.from_files(f"{input_path}/**/*.parquet")
        .flat_map(load_parquet)
        .flat_map(row_to_doc)
        .write_parquet(f"{output_path}/data-{{shard:05d}}-of-{{total:05d}}.parquet", skip_existing=True)
    )
    ctx = ZephyrContext(name="numinamath-v1-5-transform", resources=ResourceConfig(cpu=1, ram="8g"))
    ctx.execute(pipeline)


def download_numinamath_v1_5_step() -> StepSpec:
    """Download and transform valid NuminaMath-1.5 train rows into transcript documents."""
    dl = download_hf_step(
        "raw/numinamath-1.5",
        hf_dataset_id=HF_DATASET_ID,
        revision=HF_REVISION,
        hf_urls_glob=[TRAIN_PARQUET_GLOB],
    )

    return StepSpec(
        name="processed/numinamath-1.5",
        deps=[dl],
        fn=lambda output_path: transform(
            input_path=dl.output_path,
            output_path=output_path,
        ),
        hash_attrs={"version": "v1"},
    )


def numinamath_v1_5_normalize_steps() -> tuple[StepSpec, ...]:
    """Return the full ``(download+transform, normalize)`` chain for NuminaMath-1.5."""
    processed = download_numinamath_v1_5_step()
    return (
        processed,
        normalize_step(name="normalized/numinamath-1.5", download=processed),
    )
