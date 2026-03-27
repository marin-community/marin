# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Datakit canonical pipeline for FineWeb-Edu.

HuggingFace: HuggingFaceFW/fineweb-edu

FineWeb-Edu is a filtered subset of FineWeb selected for educational content.
The raw download is Parquet with columns: text, id, url, dump, file_path,
language, language_score, token_count, score, int_score.

Subsets available on HuggingFace:
- data/          — full dataset
- sample/10BT    — 10B token sample
- sample/100BT   — 100B token sample
- sample/350BT   — 350B token sample
"""

from marin.datakit.download.huggingface import download_hf_step
from marin.datakit.normalize import normalize_step
from marin.execution.step_spec import StepSpec

HF_DATASET_ID = "HuggingFaceFW/fineweb-edu"
HF_REVISION = "87f0914"


def download() -> StepSpec:
    """Download FineWeb-Edu from HuggingFace."""
    return download_hf_step(
        "raw/fineweb-edu",
        hf_dataset_id=HF_DATASET_ID,
        revision=HF_REVISION,
        override_output_path=f"raw/fineweb-edu-{HF_REVISION}",
    )


def normalize(
    *,
    subset: str = "data",
    target_partition_bytes: int = 256 * 1024 * 1024,
    override_output_path: str | None = None,
) -> StepSpec:
    """Normalize FineWeb-Edu to the datakit standard Parquet format.

    Args:
        subset: Which subset to normalize (``"data"``, ``"sample/10BT"``,
            ``"sample/100BT"``, ``"sample/350BT"``).
        target_partition_bytes: Target size per output partition.
        override_output_path: Override the computed output path.
    """
    dl = download()

    return normalize_step(
        name=f"fineweb-edu/{subset}/normalize",
        download=dl,
        input_path=f"{dl.output_path}/{HF_REVISION}/{subset}",
        target_partition_bytes=target_partition_bytes,
        override_output_path=override_output_path,
    )
