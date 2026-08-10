# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SWE-rebench ConTree trace dataset.

Line-by-line Python execution traces for the test suites of SWE-rebench V2
instances, produced by the SWE-rebench ConTree tracing pipeline and published to
the HuggingFace Hub as ``marin-community/swe-rebench-v2-CodeWorldModeling``. Each
parquet row is one (instance_id, test_id) tuple with a pre-rendered ``text`` field
containing the test source, pre-patch trace, patch, post-patch trace, and the
broad-phase trace block. The published shards are already sentinel-filtered.

Trace data lives under ``data/*.parquet``; the ``metadata/licenses.parquet``
sidecar is excluded by the download glob.
"""

from fray.types import ResourceConfig
from zephyr import counters
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext

from marin.datakit.download.huggingface import download_hf_step
from marin.datakit.download.rollout_transforms import load_parquet_batched, text_document
from marin.datakit.normalize import normalize_step
from marin.execution.step_spec import StepSpec

HF_DATASET_ID = "marin-community/swe-rebench-v2-CodeWorldModeling"
HF_REVISION = "515d954708bece40e983f71d131e0d211327adba"
SOURCE_LABEL = "marin-community/swe-rebench-v2-CodeWorldModeling"


def row_to_doc(row: dict) -> list[dict]:
    counters.pipeline.update_counter("swe_rebench_contree/kept", 1)
    return [text_document(row["text"], SOURCE_LABEL)]


def transform(input_path: str, output_path: str) -> None:
    pipeline = (
        Dataset.from_files(f"{input_path}/**/*.parquet")
        .flat_map(load_parquet_batched)
        .flat_map(row_to_doc)
        .write_parquet(f"{output_path}/data-{{shard:05d}}-of-{{total:05d}}.parquet", skip_existing=True)
    )
    ctx = ZephyrContext(name="swe-rebench-contree-transform", resources=ResourceConfig(cpu=1, ram="32g"))
    ctx.execute(pipeline)


def process_swe_rebench_contree_step() -> StepSpec:
    """Download the ConTree traces from HF and transform them into ``{id, text, source}`` docs."""
    dl = download_hf_step(
        "raw/swe-rebench-contree",
        hf_dataset_id=HF_DATASET_ID,
        revision=HF_REVISION,
        hf_urls_glob=["data/*.parquet"],
    )
    return StepSpec(
        name="processed/swe-rebench-contree",
        deps=[dl],
        fn=lambda output_path: transform(input_path=dl.output_path, output_path=output_path),
        hash_attrs={"version": "v2"},
    )


def swe_rebench_contree_normalize_steps() -> tuple[StepSpec, ...]:
    """Return the ``(download+transform, normalize)`` chain for swe-rebench-contree."""
    processed = process_swe_rebench_contree_step()
    return (
        processed,
        normalize_step(name="normalized/swe-rebench-contree", download=processed),
    )
