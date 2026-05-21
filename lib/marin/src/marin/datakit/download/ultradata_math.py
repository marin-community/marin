# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""openbmb/UltraData-Math dataset download and normalize.

UltraData-Math is organized as L1/L2/L3 quality tiers. We pin only the
text-compatible slices the Marin data-mixing plan calls for:

* ``L2-preview`` — a 33B-token preview shard of the L2 tier.
* ``L3/Conversation-Synthetic`` — conversation-styled synthetic rewrites.
* ``L3/QA-Synthetic`` — QA-styled synthetic rewrites.
* ``L3/Textbook-Exercise-Synthetic`` — textbook/exercise-styled rewrites.

L1 (170B tokens) and L3/Multi-Style-Synthetic are excluded for now to keep
the staged footprint and mixture surface small.

All parquet files share the same schema (``content`` carries the document
text, ``meta`` carries provenance JSON), so we point each subset's
normalize step at its subdirectory with ``text_field="content"``.
"""

from functools import cache

from marin.datakit.download.huggingface import download_hf_step
from marin.datakit.normalize import normalize_step
from marin.execution.step_spec import StepSpec

HF_DATASET_ID = "openbmb/UltraData-Math"
HF_REVISION = "fe10db8"

# Maps marin subset name -> path under the HF repo root.
SUBSET_DIRS: dict[str, str] = {
    "l2_preview": "data/UltraData-Math-L2-preview",
    "l3_conversation": "data/UltraData-Math-L3/Conversation-Synthetic",
    "l3_qa": "data/UltraData-Math-L3/QA-Synthetic",
    "l3_textbook_exercise": "data/UltraData-Math-L3/Textbook-Exercise-Synthetic",
}


@cache
def download_ultradata_math_step() -> StepSpec:
    """Download the selected UltraData-Math slices.

    Cached so every subset shares a single download node in the DAG. The
    glob keeps L1 and L3/Multi-Style-Synthetic out of staging.
    """
    return download_hf_step(
        "raw/ultradata_math",
        hf_dataset_id=HF_DATASET_ID,
        revision=HF_REVISION,
        hf_urls_glob=[f"{subdir}/**/*.parquet" for subdir in SUBSET_DIRS.values()] + ["*.md"],
    )


def ultradata_math_normalize_steps() -> dict[str, tuple[StepSpec, ...]]:
    """Return ``(download, normalize)`` chains for every UltraData-Math subset.

    Keyed by the registry name convention ``"ultradata_math/<subset>"``.
    All subsets put the document body in ``content`` (not ``text``); the
    overridden ``text_field`` keeps normalize's record schema consistent
    with the rest of the registry.
    """
    download = download_ultradata_math_step()
    chains: dict[str, tuple[StepSpec, ...]] = {}
    for subset, subdir in SUBSET_DIRS.items():
        marin_name = f"ultradata_math/{subset}"
        normalize = normalize_step(
            name=f"normalized/{marin_name}",
            download=download,
            text_field="content",
            id_field="id",
            relative_input_path=subdir,
            file_extensions=(".parquet",),
        )
        chains[marin_name] = (download, normalize)
    return chains
