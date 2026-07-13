# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""AlienKevin/SWE-ZERO-12M-trajectories dataset download and transform.

12.29M execution-free agentic-coding trajectories generated with the
``mini-swe-agent`` v1 format. Each row contains a multi-turn conversation
(``messages``: list of ``{role, content}``) along with the rollout's
``exit_status`` (``Submitted``, ``incomplete``, etc.).
"""

from fray.types import ResourceConfig
from zephyr import counters
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext

from marin.datakit.download.huggingface import download_hf_step
from marin.datakit.download.rollout_transforms import load_parquet_batched, render_role_message, text_document
from marin.datakit.normalize import normalize_step
from marin.execution.step_spec import StepSpec

HF_DATASET_ID = "AlienKevin/SWE-ZERO-12M-trajectories"
HF_REVISION = "44e0280"


def row_to_doc(row: dict) -> list[dict]:
    messages = row.get("messages")
    if not messages:
        counters.pipeline.update_counter("swe_zero_12m/dropped", 1)
        return []

    text = "\n\n".join(render_role_message(m) for m in messages)

    counters.pipeline.update_counter("swe_zero_12m/kept", 1)
    return [text_document(text, HF_DATASET_ID)]


def transform(input_path: str, output_path: str) -> None:
    pipeline = (
        Dataset.from_files(f"{input_path}/**/*.parquet")
        .flat_map(load_parquet_batched)
        .flat_map(row_to_doc)
        .write_parquet(f"{output_path}/data-{{shard:05d}}-of-{{total:05d}}.parquet", skip_existing=True)
    )
    ctx = ZephyrContext(name="swe-zero-12m-transform", resources=ResourceConfig(cpu=1, ram="32g"))
    ctx.execute(pipeline)


def download_swe_zero_12m_step() -> StepSpec:
    """Download SWE-ZERO-12M-trajectories and render each rollout into a Parquet text document."""
    dl = download_hf_step(
        "raw/swe-zero-12m-trajectories",
        hf_dataset_id=HF_DATASET_ID,
        revision=HF_REVISION,
    )

    return StepSpec(
        name="processed/swe-zero-12m-trajectories",
        deps=[dl],
        fn=lambda output_path: transform(
            input_path=dl.output_path,
            output_path=output_path,
        ),
        hash_attrs={"version": "v1"},
    )


def swe_zero_12m_normalize_steps() -> tuple[StepSpec, ...]:
    """Return the full ``(download+transform, normalize)`` chain for swe-zero-12m."""
    processed = download_swe_zero_12m_step()
    return (
        processed,
        normalize_step(name="normalized/swe-zero-12m", download=processed),
    )
