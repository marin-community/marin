# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""tokyotech-llm/swallow-math-v2 dataset download and normalize.

SwallowMath-v2 ships its stage3 outputs (the ready-to-tokenize math text
slices) as JSONL with a single ``text`` column. We pin the two stage3
variants — QA-style and textbook-style — and let the rest of the dataset
(intermediate stages) stay un-staged so we don't move TB of data that won't
end up in mixtures.

The download is shared across both subsets via ``@cache``; each subset gets
its own normalize step targeting its ``stage3-*`` subdirectory.
"""

from functools import cache

from marin.datakit.download.huggingface import download_hf_step
from marin.datakit.normalize import normalize_step
from marin.execution.step_spec import StepSpec

HF_DATASET_ID = "tokyotech-llm/swallow-math-v2"
HF_REVISION = "b59a686"

# Maps marin subset name -> subdirectory under the HF repo root.
SUBSET_DIRS: dict[str, str] = {
    "qa": "stage3-qa",
    "textbook": "stage3-textbook",
}


@cache
def download_swallow_math_v2_step() -> StepSpec:
    """Download the swallow-math-v2 stage3 slices.

    Cached so both stage3 subsets share a single download node in the DAG.
    The glob restricts the download to the stage3 outputs we actually mix
    (skipping the much larger intermediate stages).
    """
    return download_hf_step(
        "raw/swallow_math_v2",
        hf_dataset_id=HF_DATASET_ID,
        revision=HF_REVISION,
        hf_urls_glob=[f"{subdir}/**/*.jsonl" for subdir in SUBSET_DIRS.values()] + ["*.md"],
    )


def swallow_math_v2_normalize_steps() -> dict[str, tuple[StepSpec, ...]]:
    """Return ``(download, normalize)`` chains for every swallow-math-v2 subset.

    Keyed by the registry name convention ``"swallow_math_v2/<subset>"``.
    Both stage3 subsets store their document text under the ``text`` column,
    matching :func:`normalize_step`'s default ``text_field``.
    """
    download = download_swallow_math_v2_step()
    chains: dict[str, tuple[StepSpec, ...]] = {}
    for subset, subdir in SUBSET_DIRS.items():
        marin_name = f"swallow_math_v2/{subset}"
        normalize = normalize_step(
            name=f"normalized/{marin_name}",
            download=download,
            text_field="text",
            id_field="id",
            relative_input_path=subdir,
            file_extensions=(".jsonl",),
        )
        chains[marin_name] = (download, normalize)
    return chains
