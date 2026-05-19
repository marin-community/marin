# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SWE-rebench ConTree trace dataset.

The raw dataset is produced upstream by
``experiments/swe_rebench_trace/contree_pipeline.py`` and lives at
``$MARIN_PREFIX/raw/swe-rebench-contree-traces/``. Each parquet row is one
(instance_id, test_id) tuple with a pre-rendered ``text`` field containing
the test source, pre-patch trace, patch, post-patch trace, and the broad-phase
trace block. Rows where the pipeline gave up before any trace was captured are
written as sentinel rows with ``text=""``; we filter those here.
"""

import hashlib
import os

from fray import ResourceConfig
from rigging.filesystem import marin_prefix
from zephyr import Dataset, ZephyrContext, counters

from marin.datakit.download.rollout_transforms import load_parquet_batched
from marin.datakit.normalize import normalize_step
from marin.execution.step_spec import StepSpec

RAW_RELATIVE_PATH = "raw/swe-rebench-contree-traces"
SOURCE_LABEL = "nebius/SWE-rebench-V2 (ConTree-traced)"


def row_to_doc(row: dict) -> list[dict]:
    text = row.get("text") or ""
    if not text:
        counters.increment("swe_rebench_contree/sentinel_skipped")
        return []
    counters.increment("swe_rebench_contree/kept")
    return [
        {
            "id": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "text": text,
            "source": SOURCE_LABEL,
        }
    ]


def transform(input_path: str, output_path: str) -> None:
    pipeline = (
        Dataset.from_files(f"{input_path}/*.parquet")
        .flat_map(load_parquet_batched)
        .flat_map(row_to_doc)
        .write_parquet(f"{output_path}/data-{{shard:05d}}-of-{{total:05d}}.parquet", skip_existing=True)
    )
    ctx = ZephyrContext(name="swe-rebench-contree-transform", resources=ResourceConfig(cpu=1, ram="32g"))
    ctx.execute(pipeline)


def process_swe_rebench_contree_step() -> StepSpec:
    """Transform the staged ConTree trace parquets into ``{id, text, source}`` docs."""
    input_path = os.path.join(marin_prefix(), RAW_RELATIVE_PATH)
    return StepSpec(
        name="processed/swe-rebench-contree",
        fn=lambda output_path: transform(input_path=input_path, output_path=output_path),
        hash_attrs={"version": "v1", "source_path": input_path},
    )


def swe_rebench_contree_normalize_steps() -> tuple[StepSpec, ...]:
    """Return the ``(transform, normalize)`` chain for swe-rebench-contree."""
    processed = process_swe_rebench_contree_step()
    return (
        processed,
        normalize_step(name="normalized/swe-rebench-contree", download=processed),
    )
