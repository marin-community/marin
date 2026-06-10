# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Download + normalize helpers for bigcode/starcoder2data-extras.

Subsets: ir_cpp, ir_python, ir_rust, ir_low_resource, documentation, kaggle.
Each subset is staged under its own ``raw/starcoder2_extras-<rev>/<subset>``
path, so the downloads are per-subset (not family-shared).

The token-count-viewer advertises the repo as ``bigcode/StarCoder2-Extras``
(friendlier casing); the actual HF repo is ``bigcode/starcoder2data-extras``.
"""

from marin.datakit.download.huggingface import download_hf_step
from marin.datakit.normalize import normalize_step
from marin.execution.step_spec import StepSpec

HF_DATASET_ID = "bigcode/starcoder2data-extras"
HF_REVISION = "1ba0d4f"

SUBSETS = ["ir_cpp", "ir_python", "ir_rust", "ir_low_resource", "documentation", "kaggle"]


def download_starcoder2_extras_step(subset: str) -> StepSpec:
    """Download a single subset of the starcoder2data-extras dataset."""
    return download_hf_step(
        f"raw/starcoder2_extras/{subset}",
        hf_dataset_id=HF_DATASET_ID,
        revision=HF_REVISION,
        hf_urls_glob=[f"{subset}/*.parquet"],
        override_output_path=f"raw/starcoder2_extras-{HF_REVISION}/{subset}",
    )


def download_all_starcoder2_extras_steps() -> list[StepSpec]:
    """Download all selected subsets of starcoder2data-extras."""
    return [download_starcoder2_extras_step(subset) for subset in SUBSETS]


def starcoder2_extras_normalize_steps() -> dict[str, tuple[StepSpec, ...]]:
    """Return ``(download, normalize)`` chains for every StarCoder2-Extras subset.

    Keyed by the registry name convention ``"starcoder2/<subset>"``. The
    ``text_field="content"`` default matches StarCoder2 parquet schema (text
    lives in the ``content`` column, not ``text``).
    """
    chains: dict[str, tuple[StepSpec, ...]] = {}
    for subset in SUBSETS:
        marin_name = f"starcoder2/{subset}"
        download = download_starcoder2_extras_step(subset)
        normalize = normalize_step(
            name=f"normalized/{marin_name}",
            download=download,
            text_field="content",
            id_field="id",
            file_extensions=(".parquet",),
        )
        chains[marin_name] = (download, normalize)
    return chains
