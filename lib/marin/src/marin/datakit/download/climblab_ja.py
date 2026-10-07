# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""KantaHayashiAI/ClimbLab-Ja dataset download + normalize helpers.

~300B Japanese tokens (480 GB parquet, 201M rows) derived from LLM-jp
Corpus v4 with semantic re-clustering — a Japanese adaptation of
NVIDIA's Nemotron-ClimbLab quality-filtering pipeline. Per-document
fields ``quality``, ``advertisement``, and the four ``*_value`` scores
are preserved through normalize so downstream consumers can re-filter
without re-deriving them. License: ODC-BY.
"""

from marin.datakit.download.huggingface import download_hf_step
from marin.datakit.normalize import normalize_step
from marin.execution.step_spec import StepSpec

HF_DATASET_ID = "KantaHayashiAI/ClimbLab-Ja"
HF_REVISION = "889e349"


def climblab_ja_normalize_steps() -> tuple[StepSpec, ...]:
    """Return the ``(download, normalize)`` chain for ClimbLab-Ja."""
    download = download_hf_step(
        "raw/climblab-ja",
        hf_dataset_id=HF_DATASET_ID,
        revision=HF_REVISION,
    )
    normalize = normalize_step(
        name="normalized/climblab-ja",
        download=download,
        file_extensions=(".parquet",),
    )
    return (download, normalize)
