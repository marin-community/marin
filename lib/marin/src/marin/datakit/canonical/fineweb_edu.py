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
from marin.execution.step_spec import StepSpec


def download(
    *,
    revision: str = "87f0914",
    hf_urls_glob: list[str] | None = None,
) -> StepSpec:
    """Download FineWeb-Edu from HuggingFace."""
    return download_hf_step(
        "raw/fineweb-edu",
        hf_dataset_id="HuggingFaceFW/fineweb-edu",
        revision=revision,
        hf_urls_glob=hf_urls_glob,
        override_output_path=f"raw/fineweb-edu-{revision}",
    )
