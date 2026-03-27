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


def download(
    *,
    hf_dataset_id: str = "HuggingFaceFW/fineweb-edu",
    revision: str = "87f0914",
) -> StepSpec:
    """Download FineWeb-Edu from HuggingFace."""
    return download_hf_step(
        "raw/fineweb-edu",
        hf_dataset_id=hf_dataset_id,
        revision=revision,
        override_output_path=f"raw/fineweb-edu-{revision}",
    )


def normalize(
    dl: StepSpec | None = None,
    *,
    subset: str = "data",
    target_partition_bytes: int = 256 * 1024 * 1024,
    override_output_path: str | None = None,
) -> StepSpec:
    """Normalize FineWeb-Edu to the datakit standard Parquet format.

    Args:
        dl: Download step. Defaults to ``download()``.
        subset: Which subset to normalize (``"data"``, ``"sample/10BT"``,
            ``"sample/100BT"``, ``"sample/350BT"``).
        target_partition_bytes: Target size per output partition.
        override_output_path: Override the computed output path.
    """
    if dl is None:
        dl = download()

    return normalize_step(
        name="normalized/fineweb_edu",
        download=dl,
        input_path=f"{dl.output_path}/{subset}",
        target_partition_bytes=target_partition_bytes,
        override_output_path=override_output_path,
    )
