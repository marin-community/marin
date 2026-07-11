# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""allenai/dolma3.5_pool download + normalize helpers.

The upstream is a single HF repo with one top-level directory per component
(``cc_all_dressed``, ``dolma4pdfs``, ``dolma_code``, ``the-stack-v2``, ...);
each directory holds gzip-free ``.jsonl.zst`` shards with a plain ``text``
field. We register three components: ``dolma4pdfs`` (OCR'd PDFs), ``dolma_code``
(re-processed Stack v2 code), and ``dolma_code_prose`` (web documents mixing
prose and code). Each subset is downloaded independently (via ``hf_urls_glob``
scoped to its directory) since the full pool is nearly 10T tokens.
"""

from marin.datakit.download.huggingface import download_hf_step
from marin.datakit.normalize import normalize_step
from marin.execution.step_spec import StepSpec

HF_DATASET_ID = "allenai/dolma3.5_pool"
HF_REVISION = "d2bf6ae"

SUBSETS = ["dolma4pdfs", "dolma_code", "dolma_code_prose"]


def download_dolma3_5_pool_step(subset: str) -> StepSpec:
    """Download a single top-level component of the dolma3.5_pool dataset."""
    return download_hf_step(
        f"raw/dolma3_5_pool/{subset}",
        hf_dataset_id=HF_DATASET_ID,
        revision=HF_REVISION,
        hf_urls_glob=[f"{subset}/**/*.jsonl.zst"],
        override_output_path=f"raw/dolma3_5_pool-{HF_REVISION}/{subset}",
    )


def dolma3_5_pool_normalize_steps() -> dict[str, tuple[StepSpec, ...]]:
    """Return ``(download, normalize)`` chains for every registered dolma3.5_pool subset.

    Keyed by the registry name convention, which matches the upstream config
    name directly (``"dolma4pdfs"``, ``"dolma_code"``, ``"dolma_code_prose"``).
    """
    chains: dict[str, tuple[StepSpec, ...]] = {}
    for subset in SUBSETS:
        download = download_dolma3_5_pool_step(subset)
        normalize = normalize_step(
            name=f"normalized/{subset}",
            download=download,
            file_extensions=(".jsonl.zst",),
        )
        chains[subset] = (download, normalize)
    return chains
