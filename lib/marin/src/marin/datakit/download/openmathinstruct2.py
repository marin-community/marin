# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""nvidia/OpenMathInstruct-2 dataset download and transform.

OpenMathInstruct-2 is a synthetic math reasoning corpus derived from GSM8K,
MATH, and augmented variants. This transform materializes the full train split
as tagged transcript documents and preserves source metadata for downstream
contamination and mixture analysis.
"""

import hashlib

from fray import ResourceConfig
from zephyr import Dataset, ZephyrContext, counters, load_parquet

from marin.datakit.download.huggingface import download_hf_step
from marin.datakit.normalize import normalize_step
from marin.execution.step_spec import StepSpec

HF_DATASET_ID = "nvidia/OpenMathInstruct-2"
HF_REVISION = "469216e"
TRAIN_PARQUET_GLOB = "data/train-*.parquet"
OPENMATHINSTRUCT2_ROUGH_TOKENS_B = 4.0
EXPECTED_PROBLEM_SOURCES = frozenset({"math", "gsm8k", "augmented_math", "augmented_gsm8k"})
LONG_PROBLEM_CHARS = 1_376
LONG_SOLUTION_CHARS = 5_237


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


def row_to_doc(row: dict) -> list[dict]:
    problem = _clean_text(row, "problem")
    if problem is None:
        counters.increment("openmathinstruct2/dropped_empty_problem")
        return []

    solution = _clean_text(row, "generated_solution")
    if solution is None:
        counters.increment("openmathinstruct2/dropped_empty_solution")
        return []

    problem_source = _optional_text(row, "problem_source")
    if problem_source not in EXPECTED_PROBLEM_SOURCES:
        counters.increment("openmathinstruct2/dropped_unknown_problem_source")
        return []

    expected_answer = _optional_text(row, "expected_answer")
    if not expected_answer:
        counters.increment("openmathinstruct2/empty_expected_answer")

    if len(problem) > LONG_PROBLEM_CHARS:
        counters.increment("openmathinstruct2/long_problem")
    if len(solution) > LONG_SOLUTION_CHARS:
        counters.increment("openmathinstruct2/long_solution")

    text = f"<user>\n{problem}\n</user>\n\n<assistant>\n{solution}\n</assistant>"

    counters.increment("openmathinstruct2/kept")
    counters.increment(f"openmathinstruct2/source/{problem_source}")
    return [
        {
            "id": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "problem_hash": hashlib.sha256(problem.encode("utf-8")).hexdigest(),
            "text": text,
            "source": HF_DATASET_ID,
            "problem_source": problem_source,
            "expected_answer": expected_answer,
            "synthetic": True,
            "benchmark_adjacent": True,
            "hf_revision": HF_REVISION,
            "split": "train",
        }
    ]


def transform(input_path: str, output_path: str) -> None:
    pipeline = (
        Dataset.from_files(f"{input_path}/**/*.parquet")
        .flat_map(load_parquet)
        .flat_map(row_to_doc)
        .write_parquet(f"{output_path}/data-{{shard:05d}}-of-{{total:05d}}.parquet", skip_existing=True)
    )
    ctx = ZephyrContext(name="openmathinstruct2-transform", resources=ResourceConfig(cpu=1, ram="8g"))
    ctx.execute(pipeline)


def download_openmathinstruct2_step() -> StepSpec:
    """Download and transform the full OpenMathInstruct-2 train split."""
    dl = download_hf_step(
        "raw/openmathinstruct2",
        hf_dataset_id=HF_DATASET_ID,
        revision=HF_REVISION,
        hf_urls_glob=[TRAIN_PARQUET_GLOB],
    )

    return StepSpec(
        name="processed/openmathinstruct2",
        deps=[dl],
        fn=lambda output_path: transform(
            input_path=dl.output_path,
            output_path=output_path,
        ),
        hash_attrs={"version": "v1", "split": "train"},
    )


def openmathinstruct2_normalize_steps() -> tuple[StepSpec, ...]:
    """Return the full ``(download+transform, normalize)`` chain for OpenMathInstruct-2."""
    processed = download_openmathinstruct2_step()
    return (
        processed,
        normalize_step(name="normalized/openmathinstruct2", download=processed),
    )
